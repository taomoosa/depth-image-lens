"""Synthetic gap hypotheses do not overwrite measured decisions or widths."""
import json
import unittest

import numpy as np

from depth_image_lens import ObjectOnlyGraspDetector, TopEntryBridgeDetector


def fixture():
    shape = (64, 96)
    cloth = np.zeros(shape, bool)
    cloth[22:30, 16:78] = True
    arms = np.zeros(shape, bool)
    arms[:22, 16:26] = True
    arms[:22, 68:78] = True
    union = cloth | arms
    z = np.where(union, .95, 1.5)
    delta = np.where(union, .05, 0.)
    return z, delta, cloth, arms


def evaluate(z, delta, cloth, arms, *, gap_px=3, missing_mask=None, supply=True,
             use_depth_cut=True, width=60, rows=3, options=None, **common):
    options = options or {}
    union = cloth | arms
    sensor = np.isfinite(z) & (z > 0)
    cloth_mask = cloth & sensor if use_depth_cut else cloth
    top_mask = union & sensor if use_depth_cut else union
    support = cloth if supply else None
    if missing_mask is not None:
        support = missing_mask
    a = ObjectOnlyGraspDetector(
        min_width_px=width, min_width_rows=rows, invalid_gap_max_px=gap_px, **options
    ).detect(z, cloth_mask, depth_unit='m', difference=delta, diagnostics=True,
             gripper_masks={'left': arms}, gap_support_mask=support, **common)
    b = TopEntryBridgeDetector(
        gap_radius_px=0, invalid_gap_max_px=gap_px, **options
    ).detect_mask(z, top_mask, depth_unit='m', diagnostics=True,
                  gap_support_mask=support, **common)
    return a, b


def extra(result):
    return result.metrics['invalid_depth_gap']


class ObliqueGapTests(unittest.TestCase):
    def test_vertical_gap_keeps_observed_width_and_bridge_independent(self):
        z, delta, cloth, arms = fixture()
        z[:, 46:49] = 0
        for use_depth_cut in (True, False):
            with self.subTest(use_depth_cut=use_depth_cut):
                args = (z, delta, cloth, arms)
                observed = evaluate(*args, gap_px=0, use_depth_cut=use_depth_cut)
                repaired = evaluate(*args, use_depth_cut=use_depth_cut)
                g, bridge = repaired
                self.assertEqual((g.held, g.bilateral, bridge.bridge), ('held', 'unknown', 'unknown'))
                self.assertEqual(g.metrics['visible_width_px'], 30)
                self.assertEqual(extra(g)['candidate_sustained_width_px'], 62)
                for before, after in zip(observed, repaired):
                    self.assertEqual({k:v for k,v in after.metrics.items()
                                      if k != 'invalid_depth_gap'}, before.metrics)
                    self.assertTrue(extra(after)['candidate'])
                    self.assertEqual(extra(after)['inferred_gap_pixels'], 24)
                    inferred = after.diagnostics['invalid_depth_gap_inferred_mask']
                    self.assertFalse(inferred[z > 0].any())
                    self.assertFalse(inferred[~cloth].any())
                    json.dumps(after.to_dict(), allow_nan=False)
                np.testing.assert_array_equal(g.diagnostics['elevated_mask'],
                                              observed[0].diagnostics['elevated_mask'])
                self.assertFalse(bridge.diagnostics['observed_path_mask'].any())
                self.assertEqual(bridge.candidate_bridge, not use_depth_cut)

    def test_diagonal_offsets_and_unit_wrapper(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                z, delta, cloth, arms = fixture()
                for y in range(22, 30):
                    start = 46 + sign*(y-22)
                    z[y, start:start+2] = np.nan
                a, b = evaluate(z, delta, cloth, arms)
                self.assertTrue(extra(a)['candidate'])
                self.assertTrue(extra(b)['candidate'])
                self.assertGreater(extra(a)['inferred_gap_pixels'], 0)
                self.assertLessEqual(extra(a)['inferred_gap_pixels'], 16)
                wrapped = ObjectOnlyGraspDetector(min_width_px=60, invalid_gap_max_px=3).detect_16uc1(
                    np.where(np.isfinite(z), z*1000, 0).astype(np.uint16), cloth & np.isfinite(z),
                    difference=(delta*1000).astype(np.uint16), gap_support_mask=cloth)
                self.assertEqual(extra(wrapped)['candidate_sustained_width_px'], 62)

    def test_horizontal_cut_can_complete_required_width_rows(self):
        shape = (60, 90)
        cloth = np.zeros(shape, bool); cloth[20:27, 13:78] = True
        z = np.where(cloth, .95, 1.5); z[23:24, :] = 0.
        r = ObjectOnlyGraspDetector(min_width_rows=6, invalid_gap_max_px=1).detect(
            z, cloth & (z > 0), depth_unit='m', difference=np.where(cloth, .05, 0),
            gap_support_mask=cloth, diagnostics=True)
        self.assertEqual(r.bilateral, 'unknown')
        self.assertTrue(extra(r)['candidate'])
        self.assertEqual(extra(r)['gap_direction'], 'y')
        self.assertEqual(extra(r)['candidate_sustained_width_px'], 65)
        self.assertEqual(extra(r)['inferred_gap_pixels'], 65)

    def test_width_and_orthogonal_line_limits(self):
        for width in (1, 2, 3, 4):
            z, delta, cloth, arms = fixture(); z[:, 46:46+width] = 0.
            for r in evaluate(z, delta, cloth, arms):
                self.assertEqual(extra(r)['candidate'], width <= 3)
        z, delta, cloth, arms = fixture(); z[:, 46:49] = 0.
        support = cloth.copy(); support[24:, 46:49] = False
        for r in evaluate(z, delta, cloth, arms, missing_mask=support):
            self.assertFalse(extra(r)['candidate'])
        support = cloth.copy(); support[23:29:2, 46:49] = False
        for r in evaluate(z, delta, cloth, arms, missing_mask=support):
            self.assertFalse(extra(r)['candidate'])

    def test_valid_background_or_valid_but_masked_depth_is_not_repaired(self):
        for value in (1.5, .95):
            z, delta, cloth, arms = fixture()
            z[:, 46:49] = value
            union = (cloth | arms).copy(); union[:, 46:49] = False
            g = ObjectOnlyGraspDetector(invalid_gap_max_px=3).detect(
                z, union & ~arms, difference=delta, depth_unit='m', gap_support_mask=cloth)
            b = TopEntryBridgeDetector(invalid_gap_max_px=3).detect_mask(
                z, union, depth_unit='m', gap_support_mask=cloth)
            self.assertFalse(extra(g)['candidate'])
            self.assertFalse(extra(b)['candidate'])

    def test_optional_support_and_uncertain_excluded(self):
        z, delta, cloth, arms = fixture(); z[:, 46:49] = 0.
        args = (z, delta, cloth, arms)
        for r in evaluate(*args, supply=False):
            self.assertTrue(extra(r)['candidate'])
        for r in evaluate(*args, missing_mask=np.zeros_like(cloth)):
            self.assertFalse(extra(r)['candidate'])
        strip = np.zeros_like(cloth); strip[:, 46:49] = True
        for key, value in [('observed_mask', ~strip), ('exclude_mask', strip),
                           ('uncertain_mask', strip), ('uncertain_mask', strip | arms)]:
            for r in evaluate(*args, **{key: value}):
                self.assertFalse(extra(r)['candidate'])
        g = ObjectOnlyGraspDetector(invalid_gap_max_px=3).detect(
            z, cloth, depth_unit='m', difference=delta, background_valid=~strip,
            gap_support_mask=cloth)
        self.assertFalse(extra(g)['candidate'])

    def test_depth_mismatch_weak_and_wholly_invalid_support(self):
        for kind in ('not_lifted', 'no_measured_anchor', 'wholly_invalid'):
            z, delta, cloth, arms = fixture()
            z[:, 46:49] = 0
            if kind == 'not_lifted':
                delta[:, 49:] = 0.
            elif kind == 'no_measured_anchor':
                z[:22, 68:78] = 0.
            else:
                z[cloth | arms] = 0.
            g, b = evaluate(z, delta, cloth, arms)
            if kind != 'no_measured_anchor':
                self.assertFalse(extra(g)['candidate'])
            if kind != 'not_lifted':
                self.assertFalse(extra(b)['candidate'])

    def test_no_mask_and_large_fold_step_with_per_side_slope(self):
        z, delta, cloth, arms = fixture()
        yy, xx = np.indices(z.shape)
        # Two real surfaces, not an interpolable depth ramp across the missing band.
        z[cloth | arms] += (.003*xx+.002*yy+.25*(xx >= 47))[cloth | arms]
        for y, x, width in zip(range(22,30), [46,46,47,47,46,45,45,46], [2,2,3,2,2,3,2,2]):
            z[y,x:x+width] = np.nan
        base = evaluate(z, delta, cloth, arms, gap_px=0, supply=False)
        coupled = evaluate(z, delta, cloth, arms, supply=False)
        for before, after in zip(base,coupled):
            self.assertTrue(extra(after)['candidate'])
            self.assertGreater(extra(after)['median_cross_gap_depth_step_m'], .25)
            self.assertGreaterEqual(extra(after)['supporting_lines'], 3)
            self.assertLessEqual(extra(after)['max_missing_run_px'], 3)
            for name, original in before.diagnostics.items():
                np.testing.assert_equal(after.diagnostics[name], original)
            self.assertEqual({k:v for k,v in after.metrics.items() if k != 'invalid_depth_gap'}, before.metrics)
        a, b = coupled
        self.assertEqual(a.bilateral,'unknown')
        self.assertEqual(b.bridge,'unknown')
        self.assertEqual(extra(a)['candidate_sustained_width_px'],62)
        self.assertTrue(np.isnan(z[22,46]))  # do not publish or mutate depth values
        for r in evaluate(z, delta, cloth, arms, supply=False,
                          options={'invalid_gap_depth_jump_m':.02}):
            self.assertFalse(extra(r)['candidate'])  # optional explicit old cap

    def test_incoherent_outward_and_neighbor_line_flanks_rejected(self):
        for kind in ('outward','across_lines'):
            z, delta, cloth, arms = fixture()
            z[:,46:49] = 0.
            if kind == 'outward':
                z[22:30,44] += .03
            else:
                z[22:30:2,16:46] += .03
            for r in evaluate(z, delta, cloth, arms, supply=False):
                self.assertFalse(extra(r)['candidate'])

    def test_no_object_reference_and_asymmetric_flanks(self):
        z, delta, cloth, arms = fixture()
        z[:,46:49] = 0.
        g = ObjectOnlyGraspDetector(invalid_gap_max_px=3).detect(
            z, cloth, depth_unit='m', diagnostics=True)
        self.assertEqual(g.held,'unknown')
        self.assertFalse(extra(g)['candidate'])
        # The right attachment is present but the two cloth flanks are
        # vertically misaligned, with only two common scanlines.
        cloth[22:28,49:68] = False
        cloth[30:36,49:78] = True
        z = np.where(cloth | arms,.95,1.5)
        z[:,46:49] = 0.
        delta = np.where(cloth | arms,.05,0.)
        for r in evaluate(z,delta,cloth,arms,supply=False):
            self.assertFalse(extra(r)['candidate'])

    def test_straight_close_grippers_missing_air_do_not_form_bridge(self):
        silhouette = np.zeros((64,96),bool)
        silhouette[:35,16:26] = True
        silhouette[:35,28:38] = True
        z = np.where(silhouette,.95,1.5)
        z[11:35,26:28] = 0.
        r = TopEntryBridgeDetector(invalid_gap_max_px=3).detect_mask(
            z, silhouette, depth_unit='m', diagnostics=True)
        self.assertEqual(r.bridge,'unknown')
        self.assertFalse(extra(r)['candidate'])

    def test_identical_invalid_gap_cannot_distinguish_fold_from_separate_pieces(self):
        z, delta, cloth, arms = fixture()
        z[:,46:49] = 0.
        disjoint_cloth = cloth & (z > 0)
        # Distinct underlying descriptions but identical measurements and masks.
        folded = evaluate(z,delta,cloth,arms,supply=False)
        separated = evaluate(z,delta,disjoint_cloth,arms,supply=False)
        for a,b in zip(folded,separated):
            self.assertEqual(extra(a),extra(b))
            self.assertTrue(extra(a)['candidate'])
        self.assertEqual(separated[0].bilateral,'unknown')
        self.assertEqual(separated[1].bridge,'unknown')

    def test_no_multifragment_chain_or_unsupported_gripper_connector(self):
        z, delta, cloth, arms = fixture()
        z[:, 38:40] = 0.; z[:, 52:54] = 0.
        for r in evaluate(z, delta, cloth, arms):
            self.assertFalse(extra(r)['candidate'])
        z, delta, cloth, arms = fixture()
        z[:, 46:49] = 0.
        for r in evaluate(z, delta, cloth, arms, missing_mask=arms):
            self.assertFalse(extra(r)['candidate'])

    def test_wandering_and_variable_width_stripes(self):
        cases = (
            ('steps', [46,46,47,47,46,45,45,46], [2,2,3,2,2,3,2,2], True, True),
            ('wander', [45,46,48,49,48,46,45,44], [3,3,3,3,3,3,3,3], True, True),
            ('mixed-width', [45,46,48,49,48,46,45,44], [4,4,3,3,3,4,4,4], True, True),
            ('over-limit', [45,46,48,49,48,46,45,44], [4,4,4,4,4,4,4,4], False, False),
            ('no consecutive flanks', [45,46,48,49,48,46,45,44], [4,3,4,3,4,3,4,3], False, False),
            ('thin observed diagonal connection', [46,48,47,45,46,48,47,45],
             [3,2,3,3,2,3,3,3], True, False),
        )
        for name, starts, widths, grasp, bridge in cases:
            with self.subTest(name=name):
                z, delta, cloth, arms = fixture()
                for y, x, width in zip(range(22, 30), starts, widths):
                    z[y, x:x+width] = 0
                a, b = evaluate(z, delta, cloth, arms)
                self.assertEqual((extra(a)['candidate'], extra(b)['candidate']), (grasp,bridge))
                self.assertEqual((a.bilateral, b.bridge), ('unknown','unknown'))

    def test_one_horizontal_cut_in_u_leg_but_not_two(self):
        shape = (75, 100)
        cloth = np.zeros(shape, bool)
        cloth[15:47,16:26] = True
        cloth[15:47,68:78] = True
        cloth[38:47,16:78] = True
        arms = np.zeros(shape, bool)
        arms[:15,16:26] = True
        arms[:15,68:78] = True
        z = np.where(cloth | arms, .95, 1.5)
        delta = np.where(cloth | arms, .05, 0)
        z[28:30,68:78] = 0
        _, b = evaluate(z, delta, cloth, arms)
        self.assertEqual(b.bridge,'unknown')
        self.assertEqual(extra(b)['gap_direction'],'y')
        self.assertTrue(extra(b)['candidate'])
        z[28:30,16:26] = 0
        _, b = evaluate(z, delta, cloth, arms)
        self.assertEqual(b.bridge,'unknown')
        self.assertFalse(extra(b)['candidate'])

    def test_real_observed_bridge_and_width_and_invalid_options(self):
        z, delta, cloth, arms = fixture()
        a, b = evaluate(z, delta, cloth, arms)
        self.assertEqual((a.bilateral,b.bridge), ('bilateral','yes'))
        self.assertEqual(extra(a)['status'], 'already_observed_bilateral')
        self.assertEqual(extra(b)['status'], 'already_observed_bridge')
        for cls in (ObjectOnlyGraspDetector, TopEntryBridgeDetector):
            for kwargs in (dict(invalid_gap_max_px=True), dict(invalid_gap_max_px=9),
                           dict(invalid_gap_depth_jump_m=float('nan')), dict(invalid_gap_min_lines=1)):
                with self.assertRaises(ValueError):
                    cls(**kwargs)


if __name__ == '__main__':
    unittest.main()
