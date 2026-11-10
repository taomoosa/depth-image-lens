"""Evidence gates and deliberate counterexamples; not real-image accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector, OverheadPickCandidateDetector
from .test_rgbd_background import scene, detect


class RGBDFusionTests(unittest.TestCase):
    def run_modes(self, c, z, bc, bz, **kwargs):
        return [detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode=mode),
                       diagnostics=True, **kwargs) for mode in ('union', 'evidence')]

    def test_background_color_suppresses_weak_and_isolated_depth(self):
        c, z, bc, bz = scene()
        z[5:15, 5:20] = 993  # Weak but coherent, no color support.
        z[25, 30] = 900  # Strong but isolated.
        old, new = self.run_modes(c, z, bc, bz)
        self.assertEqual(old.object_mask.sum(), 151)
        self.assertFalse(new.object_mask.any())
        self.assertEqual(new.stages['fusion']['status'], 'unresolved')
        np.testing.assert_array_equal(new.suppressed_mask, old.object_mask)
        self.assertTrue(new.uncertain_mask[new.suppressed_mask].all())
        self.assertTrue(new.observed_mask.all())  # Sensor availability is unchanged.
        self.assertFalse(new.outline_observed_mask[new.suppressed_mask].any())
        self.assertTrue((new.diagnostics['fusion_decision'][new.suppressed_mask] == 4).all())

    def test_weak_depth_confirmed_by_color(self):
        c, z, bc, bz = scene()
        z[5:15, 5:20] = 993
        c[5:15, 5:20] = (30, 150, 210)
        old, new = self.run_modes(c, z, bc, bz)
        np.testing.assert_array_equal(new.object_mask, old.object_mask)
        self.assertTrue((new.diagnostics['fusion_decision'][new.object_mask] == 1).all())
        self.assertFalse(new.uncertain_mask[new.object_mask].any())

    def test_strong_same_color_preserves_corners_holes_and_components(self):
        c, z, bc, bz = scene()
        z[3:29, 3:27] = 970
        z[3:12, 17:27] = 1000
        z[15:22, 8:15] = 1000
        z[28:35, 42:51] = 970
        old, new = self.run_modes(c, z, bc, bz)
        np.testing.assert_array_equal(new.object_mask, old.object_mask)
        self.assertTrue(new.uncertain_mask[new.object_mask].all())
        a = OuterContourDetector().detect(old.object_mask)
        b = OuterContourDetector().detect(new.object_mask, observed_mask=new.outline_observed_mask)
        self.assertEqual(a.contours, b.contours)
        self.assertEqual(len(b.holes), 1)
        self.assertEqual(len(b.contours), 2)
        pick = OverheadPickCandidateDetector().detect(z, b, background_depth=bz,
            registered_to_mask=True, depth_scale_m=.001,
            segmentation_uncertain_mask=new.uncertain_mask)
        self.assertFalse(pick.candidate_mask.any())  # Unconfirmed same-color shape is not trusted for pick.

    def test_thin_colored_shape_keeps_endpoints_and_nonconvex_turn(self):
        c, z, bc, bz = scene()
        mask = np.zeros(z.shape, bool)
        mask[10, 5:20] = True
        mask[10:20, 19] = True
        c[mask] = (30, 150, 210)
        z[mask] = 999
        old, new = self.run_modes(c, z, bc, bz)
        np.testing.assert_array_equal(new.object_mask, mask)
        np.testing.assert_array_equal(new.object_mask, old.object_mask)
        self.assertTrue(new.uncertain_mask[mask].all())

    def test_deliberate_loss_of_unconfirmed_thin_same_color_endpoints(self):
        c, z, bc, bz = scene()
        z[10, 5:20] = 970
        old, new = self.run_modes(c, z, bc, bz)
        self.assertEqual(old.object_mask.sum(), 15)
        self.assertEqual(new.object_mask.sum(), 13)
        self.assertTrue(new.suppressed_mask[10, 5])
        self.assertTrue(new.suppressed_mask[10, 19])
        outline = OuterContourDetector().detect(new.object_mask, observed_mask=new.outline_observed_mask)
        self.assertTrue(any(s['visibility'] != 'observed' for c in outline.contours for s in c['segments']))

    def test_luma_only_shadow_deferred_chroma_only_retained(self):
        c, z, bc, bz = scene()
        c[5:15, 5:20] = 40
        c[20:30, 30:45] = (30, 150, 210)
        old, new = self.run_modes(c, z, bc, bz)
        self.assertEqual(old.object_mask.sum(), 300)
        self.assertEqual(new.object_mask.sum(), 150)
        self.assertTrue(new.deferred_mask[5:15, 5:20].all())
        self.assertTrue(new.uncertain_mask[new.object_mask].all())

    def test_missing_depth_and_color_policies(self):
        for policy in ('color', 'unknown'):
            c, z, bc, bz = scene()
            c[5:15, 5:20] = (30, 150, 210)
            z[5:15, 5:20] = 0
            r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence', missing_depth_policy=policy))
            self.assertEqual(r.object_mask.sum(), 150 if policy == 'color' else 0)
            self.assertTrue(r.uncertain_mask[5:15, 5:20].all())
            self.assertEqual(r.observed_mask[5, 5], policy == 'color')
        c, z, bc, bz = scene()
        z[5:15, 5:20] = 970
        z[25, 30] = 993
        absent = np.zeros(z.shape, bool)
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence'), color_valid_mask=absent)
        self.assertEqual(r.object_mask.sum(), 150)
        self.assertTrue(r.deferred_mask[25, 30])
        self.assertFalse(r.suppressed_mask.any())  # Missing color cannot claim background agreement.

    def test_clipped_and_rejected_color_cannot_confirm_or_suppress(self):
        c, z, bc, bz = scene()
        c[:] = 255
        z[5:15, 5:20] = 993
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence'))
        self.assertFalse(r.object_mask.any())
        self.assertFalse(r.suppressed_mask.any())
        check = np.zeros(z.shape, bool); check[:5] = True
        c[:] = (30, 150, 210)
        z[20:30, 30:45] = 970
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence'), background_check_mask=check)
        self.assertEqual(r.stages['color_reference']['status'], 'rejected')
        self.assertEqual(r.object_mask.sum(), 150)
        self.assertTrue(r.deferred_mask[5:15, 5:20].all())
        self.assertFalse(r.suppressed_mask.any())

    def test_farther_veto_and_roi(self):
        c, z, bc, bz = scene()
        c[5:15, 5:20] = (30, 150, 210)
        z[5:15, 5:20] = 1100
        roi = np.ones(z.shape, bool); roi[:, :10] = False
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence'), work_roi=roi)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.conflict_mask[5:15, 10:20].all())
        self.assertTrue(r.deferred_mask[5:15, 10:20].all())
        for mask in (r.deferred_mask, r.suppressed_mask, r.uncertain_mask, r.observed_mask):
            self.assertFalse(mask[~roi].any())

    def test_correlated_false_signals_and_color_shift_remain_limitations(self):
        c, z, bc, bz = scene()
        c[5, 5] = (30, 150, 210); z[5, 5] = 993
        c[10:20, 10:20] = 40; z[10:20, 10:20] = 993
        r = self.run_modes(c, z, bc, bz)[1]
        self.assertTrue(r.object_mask[5, 5])  # Independent-looking but correlated noise passes.
        self.assertTrue(r.object_mask[10:20, 10:20].all())  # Shadow + depth bias passes.
        bc[25:35, 35:45] = (30, 150, 210)
        c[25:35, 37:47] = (30, 150, 210)  # 2px registration error.
        r = self.run_modes(c, z, bc, bz)[1]
        self.assertTrue(r.object_mask[25:35, 35:37].all())  # Chroma artifacts are unresolved candidates.
        self.assertTrue(r.uncertain_mask[25:35, 35:37].all())

    def test_strong_coherent_depth_bias_cannot_be_distinguished(self):
        c, z, bc, bz = scene()
        z[5:15, 5:20] = 970
        r = self.run_modes(c, z, bc, bz)[1]
        self.assertEqual(r.object_mask.sum(), 150)
        self.assertTrue(r.uncertain_mask[r.object_mask].all())

    def test_min_area_decision_matches_final_mask(self):
        c, z, bc, bz = scene()
        c[5, 5] = (30, 150, 210); z[5, 5] = 970
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence', min_object_area=2), diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertEqual(r.diagnostics['fusion_decision'][5, 5], 6)
        self.assertFalse(r.diagnostics['accepted_both_mask'].any())

    def test_support_exact_neighbor_count_and_tolerance(self):
        c, z, bc, bz = scene()
        z[10, 10:13] = (970, 970, 960)
        r = self.run_modes(c, z, bc, bz)[1]
        self.assertEqual(r.diagnostics['depth_support_fraction'][10, 11], 1/8)
        self.assertFalse(r.object_mask.any())
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='evidence', depth_support_fraction=.125))
        self.assertTrue(r.object_mask[10, 10:12].all())
        self.assertFalse(r.object_mask[10, 12])

    def test_small_images_large_radius_and_frame_corners(self):
        c, z, bc, bz = scene()
        z[:] = 970
        r = self.run_modes(c, z, bc, bz)[1]
        self.assertTrue(r.object_mask.all())  # Three neighbors support each frame corner.
        r = detect(c[:2, :2], z[:2, :2], bc[:2, :2], bz[:2, :2],
                   RGBDBackgroundSegmenter(fusion_mode='evidence', depth_support_radius_px=5))
        self.assertFalse(r.object_mask.any())

    def test_validation_and_legacy_thresholds(self):
        for key, values in {'strong_depth_ratio': [0, .5, True, np.nan, np.inf],
                            'depth_support_fraction': [0, -1, 1.1, True, np.nan],
                            'depth_support_radius_px': [0, 6, 1., True],
                            'depth_support_tolerance_m': [-1, True, np.nan, np.inf]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RGBDBackgroundSegmenter(**{key: value})
        RGBDBackgroundSegmenter(depth_threshold_m=.125)
        RGBDBackgroundSegmenter(depth_support_tolerance_m=0)

    def test_diagnostics_optional_units_serialization_and_no_mutation(self):
        c, z, bc, bz = scene()
        c[5:15, 5:20] = (30, 150, 210); z[5:15, 5:20] = 970
        copies = [a.copy() for a in (c, z, bc, bz)]
        d = RGBDBackgroundSegmenter(fusion_mode='evidence')
        a = detect(c, z, bc, bz, d)
        b = d.detect(c[..., ::-1], z.astype(float)/1000, background_color=bc[..., ::-1],
            background_depth=bz.astype(float)/1000, color_order='BGR', registered_to_depth=True,
            depth_scale_m=1, diagnostics=True)
        np.testing.assert_array_equal(a.object_mask, b.object_mask)
        self.assertEqual(a.diagnostics, {})
        self.assertEqual(b.render_debug(c).dtype, np.uint8)
        json.dumps(b.to_dict(), allow_nan=False)
        for orig, copy in zip((c, z, bc, bz), copies):
            np.testing.assert_array_equal(orig, copy)
        for mode in ('union', 'depth_required'):
            r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode=mode))
            self.assertFalse(r.suppressed_mask.any())
            self.assertFalse(r.deferred_mask.any())
            np.testing.assert_array_equal(r.outline_observed_mask, r.observed_mask)
