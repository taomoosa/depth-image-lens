"""Depth-anchored inference contracts and counterexamples, not sensor accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector
from .test_rgbd_background import scene, detect


class RGBDAnchoredTests(unittest.TestCase):
    def run_anchored(self, data, *, settings=None, **kwargs):
        return detect(*data, RGBDBackgroundSegmenter(
            fusion_mode='depth_anchored', **(settings or {})), **kwargs)

    def patch(self):
        c, z, bc, bz = scene()
        c[8:27, 8:32] = 45  # Luminance-only foreground, no chroma prerequisite.
        z[8:27, 8:32] = 980
        c[14:19, 17:22] = bc[14:19, 17:22]
        z[14:19, 17:22] = 1000
        return c, z, bc, bz

    def test_gray_object_hole_and_raw_evidence(self):
        data = self.patch()
        result = self.run_anchored(data, diagnostics=True)
        np.testing.assert_array_equal(result.object_mask, data[1] < data[3])
        np.testing.assert_array_equal(result.depth_foreground_mask, data[1] < data[3])
        np.testing.assert_allclose(result.diagnostics['depth_difference_m'],
                                   (data[3].astype(float) - data[1]) * .001)
        self.assertTrue(result.confident_foreground_mask.any())
        self.assertTrue(result.confident_background_mask.any())
        self.assertFalse(result.object_mask[14:19, 17:22].any())
        self.assertEqual(len(OuterContourDetector().detect(result.object_mask).holes), 1)

    def test_color_only_dark_and_colored_shadows_cannot_seed(self):
        for color in ((5, 10, 7), (45, 45, 45), (20, 70, 100)):
            c, z, bc, bz = scene()
            c[8:27, 8:32] = color
            r = self.run_anchored((c, z, bc, bz), diagnostics=True)
            self.assertFalse(r.object_mask.any())
            self.assertFalse(r.diagnostics['foreground_anchor_mask'].any())
            self.assertTrue(r.unknown_mask[8:27, 8:32].all())
            self.assertFalse(r.confident_foreground_mask.any())

    def test_same_color_depth_object_is_selected_but_not_confident(self):
        c, z, bc, bz = scene()
        z[8:27, 8:32] = 980
        r = self.run_anchored((c, z, bc, bz))
        self.assertGreater(r.object_mask.sum(), 300)
        self.assertTrue(r.uncertain_mask[r.object_mask].all())
        self.assertFalse(r.confident_foreground_mask.any())

    def test_one_pixel_lines_diagonal_endpoints_and_small_component(self):
        for color in (45, 140):
            for depth in (980, 993):
                c, z, bc, bz = scene()
                expected = np.zeros(z.shape, bool)
                expected[5, 4:23] = True
                expected[5:17, 22] = True
                expected[np.arange(20, 29), np.arange(5, 14)] = True
                # 13-pixel disk has no square 5x5 core.
                yy, xx = np.indices(z.shape)
                small = (yy-25)**2 + (xx-40)**2 <= 4
                expected |= small
                c[expected] = color
                z[expected] = depth
                r = self.run_anchored((c, z, bc, bz), diagnostics=True)
                # Weak tiny detached regions are an explicit unresolved limit.
                target = expected & (~small if depth == 993 else True)
                self.assertTrue(r.object_mask[target].all())
                self.assertLessEqual(int(r.object_mask[~expected].sum()), 1 if color == 140 else 0)
                self.assertTrue(r.unknown_mask[r.object_mask & ~expected].all())
                self.assertTrue(r.unknown_mask[target & ~small].all())

    def test_single_depth_spike_is_deferred_not_removed_from_raw(self):
        c, z, bc, bz = scene()
        z[10, 10] = 900
        r = self.run_anchored((c, z, bc, bz), diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.depth_foreground_mask[10, 10])
        self.assertTrue(r.suppressed_mask[10, 10])
        self.assertTrue(r.unknown_mask[10, 10])
        self.assertFalse(r.outline_observed_mask[10, 10])
        self.assertEqual(r.diagnostics['fusion_decision'][10, 10], 4)

    def test_missing_depth_never_filled_under_either_legacy_policy(self):
        c, z, bc, bz = self.patch()
        z[10:13, 10:14] = 0
        for policy in ('color', 'unknown'):
            r = self.run_anchored((c, z, bc, bz), settings={'missing_depth_policy': policy})
            self.assertFalse(r.object_mask[10:13, 10:14].any())
            self.assertFalse(r.observed_mask[10:13, 10:14].any())
            self.assertFalse(r.outline_observed_mask[10:13, 10:14].any())
            self.assertTrue(r.unknown_mask[10:13, 10:14].all())

    def test_background_invalid_depth_sentinel_and_farther(self):
        c, z, bc, bz = self.patch()
        bz[10:12, 10:12] = 0
        z[22:24, 11:14] = 65535
        z[22:24, 27:30] = 1100
        r = self.run_anchored((c, z, bc, bz), invalid_values=(65535,))
        for mask in ((bz == 0), (z == 65535), (z == 1100)):
            self.assertFalse(r.object_mask[mask].any())
            self.assertTrue(r.unknown_mask[mask].all())

    def test_current_edges_survive_rejected_or_invalid_reference(self):
        c, z, bc, bz = self.patch()
        check = np.zeros(z.shape, bool); check[:5] = True
        c[:5] = 200
        rejected = self.run_anchored((c, z, bc, bz), background_check_mask=check, diagnostics=True)
        self.assertEqual(rejected.stages['color_reference']['status'], 'rejected')
        self.assertTrue(rejected.diagnostics['current_color_edge_valid_mask'].all())
        self.assertFalse(rejected.diagnostics['color_usable_mask'].any())
        self.assertGreater(rejected.object_mask.sum(), 350)
        invalid = self.run_anchored((c, z, bc, bz),
            background_color_valid_mask=np.zeros(z.shape, bool), diagnostics=True)
        self.assertTrue(invalid.diagnostics['current_color_edge_valid_mask'].all())
        self.assertFalse(invalid.diagnostics['color_usable_mask'].any())
        self.assertGreater(invalid.object_mask.sum(), 350)

    def test_black_and_white_current_edges_are_not_disabled(self):
        for value in (0, 255):
            c, z, bc, bz = self.patch()
            c[z < bz] = value
            r = self.run_anchored((c, z, bc, bz), diagnostics=True)
            self.assertTrue(r.diagnostics['current_color_edge_valid_mask'].all())
            np.testing.assert_array_equal(r.object_mask, z < bz)
            self.assertTrue(r.unknown_mask[r.object_mask].all())

    def test_current_color_validity_is_respected(self):
        data = self.patch()
        invalid = np.zeros(data[1].shape, bool)
        r = self.run_anchored(data, color_valid_mask=invalid, diagnostics=True)
        self.assertFalse(r.diagnostics['current_color_edge_valid_mask'].any())
        self.assertFalse(r.confident_foreground_mask.any())
        self.assertTrue(r.unknown_mask.all())

    def test_confidence_partition_and_inference_always_unknown(self):
        r = self.run_anchored(self.patch(), diagnostics=True)
        fg, bg, u = r.confident_foreground_mask, r.confident_background_mask, r.unknown_mask
        np.testing.assert_array_equal(fg.astype(int) + bg.astype(int) + u.astype(int), 1)
        np.testing.assert_array_equal(u, r.uncertain_mask)
        self.assertTrue(r.object_mask[fg].all())
        self.assertFalse(r.object_mask[bg].any())
        self.assertTrue(r.inferred_foreground_mask.any())
        self.assertTrue(u[r.inferred_foreground_mask].all())
        self.assertTrue(r.object_mask[r.inferred_foreground_mask].all())
        self.assertFalse(r.outline_observed_mask[u & ~r.object_mask].any())

    def test_roi_cuts_outside_and_tiny_images(self):
        data = self.patch(); shape = data[1].shape
        roi = np.ones(shape, bool); roi[:, 20:23] = False
        r = self.run_anchored(data, work_roi=roi, diagnostics=True)
        for key in ('object_mask', 'uncertain_mask', 'observed_mask', 'suppressed_mask',
                    'deferred_mask', 'unknown_mask', 'confident_foreground_mask',
                    'confident_background_mask', 'inferred_foreground_mask'):
            self.assertFalse(getattr(r, key)[~roi].any())
        empty = self.run_anchored(data, work_roi=np.zeros(shape, bool))
        self.assertEqual(empty.stages['fusion']['status'], 'empty_domain')
        for n in (2, 3):
            tiny = tuple(a[:n, :n] for a in data)
            r = self.run_anchored(tiny, settings={'anchored_refine_radius_px': 20})
            self.assertFalse(r.object_mask.any())

    def test_farther_depth_is_not_crossed_by_domain_growth(self):
        c, z, bc, bz = scene()
        c[8:27, 5:15] = 40; z[8:27, 5:15] = 980
        z[:, 15] = 1100
        r = self.run_anchored((c, z, bc, bz),
                              settings={'anchored_refine_radius_px': 20}, diagnostics=True)
        self.assertFalse(r.diagnostics['refinement_domain_mask'][:, 15:].any())
        self.assertFalse(r.object_mask[:, 15:].any())

    def test_coarse_components_cannot_inherit_anchors_across_farther_barrier(self):
        bc = np.full((40, 70, 3), 140, np.uint8)
        bz = np.ones((40, 70))
        for wall_color in (40, 140):
            c, z = bc.copy(), bz.copy()
            c[:, 5:33] = 40; c[:, 30] = wall_color
            z[:, 5:30] = .980; z[:, 30] = 1.1; z[:, 31:33] = .994
            r = RGBDBackgroundSegmenter(fusion_mode='depth_anchored',
                anchored_refine_radius_px=0).detect(c, z, background_color=bc,
                    background_depth=bz, color_order='RGB', registered_to_depth=True,
                    depth_scale_m=1, diagnostics=True)
            self.assertFalse(r.diagnostics['foreground_anchor_mask'][:, 31:33].any())
            self.assertFalse(r.diagnostics['refinement_domain_mask'][:, 30:].any())
            self.assertFalse(r.object_mask[:, 30:].any())

    def test_threshold_margin_stays_unknown_and_solver_state_is_exposed(self):
        c, z, bc, bz = scene()
        c[8:27, 8:32] = 40
        z = z.astype(float); z[8:27, 8:32] = 994.5
        r = self.run_anchored((c, z, bc, bz), diagnostics=True)
        self.assertTrue(r.object_mask.any())
        self.assertFalse(r.confident_foreground_mask.any())
        self.assertTrue(r.unknown_mask[r.object_mask].all())
        self.assertIn('refinement_converged', r.metrics)
        self.assertGreaterEqual(r.metrics['refinement_max_update'], 0)
        self.assertTrue(np.isfinite(r.diagnostics['refinement_score']).all())
        self.assertTrue(((r.diagnostics['refinement_score'] >= 0) &
                         (r.diagnostics['refinement_score'] <= 1)).all())

    def test_min_area_does_not_upgrade_filtered_evidence(self):
        c, z, bc, bz = scene()
        c[10:13, 10:13] = 40; z[10:13, 10:13] = 980
        r = self.run_anchored((c, z, bc, bz), settings={'min_object_area': 20}, diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertFalse(r.inferred_foreground_mask.any())
        self.assertFalse(r.confident_foreground_mask.any())
        removed = r.diagnostics['filtered_mask']
        self.assertTrue(removed.any())
        self.assertTrue(r.unknown_mask[removed].all())
        self.assertTrue((r.diagnostics['fusion_decision'][removed] == 6).all())
        self.assertTrue(r.diagnostics['foreground_anchor_mask'].any())  # provenance retained
        self.assertEqual(sum(r.metrics['decision_counts'].values()), z.size)

    def test_one_pixel_same_color_hole_preserves_raw_counterevidence_as_unknown(self):
        c, z, bc, bz = scene()
        z[8:27, 8:32] = 980
        z[17, 20] = 1000  # An isolated but valid observed table pixel.
        r = self.run_anchored((c, z, bc, bz), diagnostics=True)
        self.assertFalse(r.object_mask[17, 20])
        self.assertTrue(r.unknown_mask[17, 20])
        self.assertFalse(r.confident_background_mask[17, 20])
        self.assertFalse(r.outline_observed_mask[17, 20])
        self.assertFalse(r.diagnostics['background_anchor_mask'][17, 20])
        self.assertTrue(r.diagnostics['raw_background_guard_mask'][17, 20])
        # Identical inputs can mean an erroneous table-depth report inside a
        # solid same-color object. The resulting false hole must stay unknown.
        self.assertEqual(r.diagnostics['fusion_decision'][17, 20], 5)
        check = np.zeros(z.shape, bool); check[:5] = True
        c[:5] = 200
        rejected = self.run_anchored((c, z, bc, bz),
            background_check_mask=check, diagnostics=True)
        self.assertFalse(rejected.diagnostics['raw_background_guard_mask'].any())
        c[:] = 255; bc[:] = 255
        clipped = self.run_anchored((c, z, bc, bz), diagnostics=True)
        self.assertFalse(clipped.diagnostics['raw_background_guard_mask'].any())

    def test_explicit_limit_coherent_shadow_can_even_look_confident(self):
        data = self.patch()  # Identical input could be a shadow plus depth bias.
        r = self.run_anchored(data)
        self.assertTrue(r.object_mask.any())
        self.assertTrue(r.confident_foreground_mask.any())
        self.assertIn('coherent_depth_bias_and_corrupted_holes_can_pass', r.stages['fusion']['warnings'])

    def test_explicit_limit_corrupted_hole_and_below_threshold_object(self):
        c, z, bc, bz = self.patch()
        z[14:19, 17:22] = 980  # A true hole with corrupted depth can be filled.
        r = self.run_anchored((c, z, bc, bz))
        self.assertTrue(r.object_mask[14:19, 17:22].all())
        self.assertTrue(r.unknown_mask[14:19, 17:22].all())
        c, z, bc, bz = self.patch(); z[z < bz] = 996
        r = self.run_anchored((c, z, bc, bz))
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.unknown_mask[z < bz].all())

    def test_old_modes_ignore_anchored_parameters_and_new_masks_are_optional(self):
        data = self.patch()
        for mode in ('union', 'depth_required', 'evidence', 'color_seeded'):
            a = detect(*data, RGBDBackgroundSegmenter(fusion_mode=mode))
            b = detect(*data, RGBDBackgroundSegmenter(fusion_mode=mode,
                anchored_color_sigma=50, anchored_pairwise_weight=0,
                anchored_refine_radius_px=0, anchored_iterations=1, anchored_anchor_ratio=3))
            for key in ('object_mask', 'uncertain_mask', 'observed_mask', 'suppressed_mask', 'deferred_mask'):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            self.assertIsNone(a.unknown_mask)

    def test_extreme_affinity_and_roundoff_scores_stay_finite_bounded(self):
        rng = np.random.default_rng(42)
        c = rng.integers(20, 200, (32, 40, 3), dtype=np.uint8)
        bc = c.copy(); c[8:25, 8:32] //= 2
        z = np.ones((32, 40)); z[8:25, 8:32] = .98
        z += rng.normal(0, .002, z.shape)
        for sigma in (1e-300, 1, 441):
            r = RGBDBackgroundSegmenter(fusion_mode='depth_anchored',
                anchored_color_sigma=sigma, anchored_pairwise_weight=100).detect(
                    c, z, background_color=bc, background_depth=np.ones(z.shape),
                    color_order='RGB', registered_to_depth=True, depth_scale_m=1,
                    diagnostics=True)
            score = r.diagnostics['refinement_score']
            self.assertTrue(np.isfinite(score).all())
            self.assertTrue(((score >= 0) & (score <= 1)).all())

    def test_validation(self):
        cases = {'anchored_color_sigma': [0, -1, 442, True, np.nan, np.inf, [1]],
                 'anchored_pairwise_weight': [-1, 101, False, np.nan, np.inf, [1]],
                 'anchored_anchor_ratio': [.9, 4.1, True, np.nan, np.inf, [1]],
                 'anchored_refine_radius_px': [-1, 21, 1., True, np.int64(2)],
                 'anchored_iterations': [0, 101, 1., True, np.int64(2)]}
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RGBDBackgroundSegmenter(**{key: value})

    def test_units_strides_no_mutation_diagnostics_json_debug(self):
        data = self.patch(); saved = [a.copy() for a in data]
        a = self.run_anchored(data)
        c, z, bc, bz = data
        b = RGBDBackgroundSegmenter(fusion_mode='depth_anchored').detect(
            c[..., ::-1], z.astype(float)/1000, background_color=bc[..., ::-1],
            background_depth=bz.astype(float)/1000, color_order='BGR', depth_scale_m=1,
            registered_to_depth=True, diagnostics=True)
        np.testing.assert_array_equal(a.object_mask, b.object_mask)
        np.testing.assert_array_equal(a.unknown_mask, b.unknown_mask)
        self.assertEqual(a.diagnostics, {})
        json.dumps(b.to_dict(), allow_nan=False)
        self.assertEqual(b.render_debug(c).dtype, np.uint8)
        self.assertEqual(b.render_debug(c[..., ::-1], color_order='BGR').dtype, np.uint8)
        for original, copy in zip(data, saved):
            np.testing.assert_array_equal(original, copy)


if __name__ == '__main__':
    unittest.main()
