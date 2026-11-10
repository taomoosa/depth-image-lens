"""Bounded color-guided inference and explicit counterexamples, not field accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector
from .test_rgbd_background import scene, detect


STRONG = (30, 150, 210)
WEAK = (110, 145, 165)  # Lab ellipse radius ~.78: below strict, above relaxed.


class RGBDSeededTests(unittest.TestCase):
    def run_seeded(self, c, z, bc, bz, *, settings=None, **kwargs):
        return detect(c, z, bc, bz, RGBDBackgroundSegmenter(
            fusion_mode='color_seeded', **(settings or {})), **kwargs)

    def patch(self):
        c, z, bc, bz = scene()
        c[8:24, 8:28] = WEAK
        c[10:22, 10:26] = STRONG
        z[8:24, 8:28] = 980
        return c, z, bc, bz

    def test_recovers_weak_band_without_smoothing_seed_boundary(self):
        c, z, bc, bz = self.patch()
        expected = z < bz
        # Narrow strict-colored tab/endpoints survive without depth support.
        c[7, 15:20] = STRONG
        expected[7, 15:20] = True
        r = self.run_seeded(c, z, bc, bz, diagnostics=True)
        np.testing.assert_array_equal(r.object_mask, expected)
        grown = r.diagnostics['accepted_growth_mask']
        self.assertTrue(grown.any())
        self.assertTrue(r.uncertain_mask[grown].all())
        self.assertTrue(r.observed_mask[grown].all())
        self.assertTrue((r.diagnostics['growth_step'][grown] > 0).all())
        self.assertTrue((r.diagnostics['fusion_decision'][grown] == 7).all())
        self.assertEqual(r.metrics['grown_pixels'], int(grown.sum()))

    def test_zero_steps_is_seed_only_and_limit_is_geodesic(self):
        c, z, bc, bz = scene()
        c[10:20, 10:35] = WEAK
        c[10:20, 10:12] = STRONG
        z[10:20, 10:35] = 980
        for steps in (0, 1, 2, 5):
            r = self.run_seeded(c, z, bc, bz, settings={'color_growth_max_steps': steps}, diagnostics=True)
            expected = np.zeros(z.shape, bool)
            expected[10:20, 10:12+steps] = True
            np.testing.assert_array_equal(r.object_mask, expected)
            self.assertEqual(int(r.diagnostics['growth_step'].max()), steps)

    def test_gap_cannot_be_bridged_and_no_wrapping(self):
        c, z, bc, bz = scene()
        c[10:20, :2] = STRONG
        c[10:20, -2:] = WEAK
        c[10:20, 3:8] = WEAK
        z[10:20, :] = 980
        r = self.run_seeded(c, z, bc, bz, settings={'color_growth_max_steps': 20})
        self.assertEqual(r.object_mask.sum(), 20)
        self.assertTrue((r.suppressed_mask | r.deferred_mask)[10:20, 3:8].all())

    def test_small_and_one_pixel_strict_features_keep_endpoints(self):
        c, z, bc, bz = scene()
        expected = np.zeros(z.shape, bool)
        expected[3, 3] = True
        expected[10, 10:25] = True
        expected[10:20, 24] = True
        c[expected] = STRONG
        r = self.run_seeded(c, z, bc, bz)
        np.testing.assert_array_equal(r.object_mask, expected)
        self.assertTrue(r.uncertain_mask[expected].all())

    def test_true_hole_and_multiple_components_not_filled_by_shape(self):
        c, z, bc, bz = self.patch()
        c[14:18, 16:20] = bc[14:18, 16:20]
        z[14:18, 16:20] = 1000
        c[28:32, 40:44] = STRONG
        z[28:32, 40:44] = 980
        r = self.run_seeded(c, z, bc, bz)
        np.testing.assert_array_equal(r.object_mask, z < bz)
        outline = OuterContourDetector().detect(r.object_mask, observed_mask=r.outline_observed_mask)
        self.assertEqual(len(outline.holes), 1)
        self.assertEqual(len(outline.contours), 2)

    def test_hole_needs_color_and_depth_not_enclosure(self):
        for weak_color, depth in ((False, 980), (True, 1000), (True, 0), (True, 1100)):
            c, z, bc, bz = self.patch()
            c[14:18, 16:20] = WEAK if weak_color else bc[14:18, 16:20]
            z[14:18, 16:20] = depth
            r = self.run_seeded(c, z, bc, bz)
            self.assertFalse(r.object_mask[14:18, 16:20].any())

    def test_supported_color_hole_repaired_as_uncertain(self):
        c, z, bc, bz = self.patch()
        c[14:18, 16:20] = WEAK
        r = self.run_seeded(c, z, bc, bz, diagnostics=True)
        self.assertTrue(r.object_mask[14:18, 16:20].all())
        self.assertTrue(r.uncertain_mask[14:18, 16:20].all())

    def test_isolated_depth_artifact_cannot_expand(self):
        c, z, bc, bz = scene()
        c[10, 10] = STRONG
        c[10, 11] = WEAK
        z[10, 11] = 900
        r = self.run_seeded(c, z, bc, bz, diagnostics=True)
        self.assertTrue(r.object_mask[10, 10])
        self.assertFalse(r.object_mask[10, 11])
        self.assertEqual(r.diagnostics['depth_support_fraction'][10, 11], 0)
        self.assertTrue((r.suppressed_mask | r.deferred_mask)[10, 11])

    def test_depth_noise_support_tolerance_and_no_strong_requirement(self):
        c, z, bc, bz = self.patch()
        z[z < bz] = 993  # Below strong_depth_ratio * T but coherent > T.
        r = self.run_seeded(c, z, bc, bz)
        np.testing.assert_array_equal(r.object_mask, z < bz)
        z[8, 12] = 970  # No neighbor with matching dZ, despite strong depth.
        r = self.run_seeded(c, z, bc, bz)
        self.assertFalse(r.object_mask[8, 12])

    def test_missing_policies_farther_veto_and_unknown_outline(self):
        c, z, bc, bz = self.patch()
        z[10:13, 10:14] = 0
        z[17:20, 18:22] = 1100
        for policy in ('color', 'unknown'):
            r = self.run_seeded(c, z, bc, bz, settings={'missing_depth_policy': policy})
            self.assertEqual(bool(r.object_mask[10:13, 10:14].all()), policy == 'color')
            self.assertEqual(bool(r.observed_mask[10:13, 10:14].all()), policy == 'color')
            self.assertTrue(r.uncertain_mask[10:13, 10:14].all())
            self.assertFalse(r.object_mask[17:20, 18:22].any())
            self.assertTrue(r.conflict_mask[17:20, 18:22].all())
            self.assertFalse(r.outline_observed_mask[17:20, 18:22].any())

    def test_clipping_missing_color_and_rejected_reference_no_fallback(self):
        for mode in ('clipped', 'invalid', 'rejected'):
            c, z, bc, bz = self.patch()
            options = {}
            if mode == 'clipped':
                c[:] = (255, 100, 20)
            elif mode == 'invalid':
                options['color_valid_mask'] = np.zeros(z.shape, bool)
            else:
                c[:5] = STRONG
                check = np.zeros(z.shape, bool); check[:5] = True
                options['background_check_mask'] = check
            r = self.run_seeded(c, z, bc, bz, **options)
            self.assertFalse(r.object_mask.any())
            self.assertTrue(r.deferred_mask[z < bz].all())
            if mode == 'rejected':
                self.assertEqual(r.stages['color_reference']['selection_action'], 'disable_color_no_seeds')

    def test_roi_cuts_are_not_bridged(self):
        c, z, bc, bz = self.patch()
        roi = np.ones(z.shape, bool); roi[8:24, 9] = False
        r = self.run_seeded(c, z, bc, bz, work_roi=roi)
        self.assertFalse(r.object_mask[:, 8:10].any())
        for name in ('object_mask', 'uncertain_mask', 'observed_mask', 'suppressed_mask', 'deferred_mask'):
            self.assertFalse(getattr(r, name)[~roi].any())

    def test_limitation_same_color_and_luma_only_have_no_seed(self):
        for color in ((140, 140, 140), (40, 40, 40)):
            c, z, bc, bz = scene()
            c[10:20, 10:20] = color
            z[10:20, 10:20] = 970
            r = self.run_seeded(c, z, bc, bz)
            self.assertFalse(r.object_mask.any())
            self.assertEqual(r.stages['fusion']['status'], 'unresolved')
            self.assertFalse(r.outline_observed_mask[10:20, 10:20].any())

    def test_limitation_adjacent_shadow_plus_depth_error_can_grow(self):
        c, z, bc, bz = scene()
        c[10:20, 10:20] = STRONG
        c[10:20, 20:30] = 40  # Pure shadow, no material in this constructed case.
        z[10:20, 10:30] = 993  # Coherent depth bias cannot be distinguished here.
        r = self.run_seeded(c, z, bc, bz)
        self.assertTrue(r.object_mask[10:20, 20:22].all())
        self.assertFalse(r.object_mask[10:20, 22:30].any())
        self.assertTrue(r.uncertain_mask[10:20, 20:22].all())

    def test_limitation_false_hole_evidence_and_colored_shift_can_pass(self):
        c, z, bc, bz = self.patch()
        c[14:18, 16:20] = WEAK  # Could be a real hole with both modalities wrong.
        r = self.run_seeded(c, z, bc, bz)
        self.assertTrue(r.object_mask[14:18, 16:20].all())
        c, z, bc, bz = scene()
        bc[10:20, 10:20] = STRONG
        c[10:20, 12:22] = STRONG
        r = self.run_seeded(c, z, bc, bz)
        self.assertEqual(r.object_mask.sum(), 40)  # Registration is not corrected.
        self.assertTrue(r.uncertain_mask[r.object_mask].all())

    def test_min_area_final_acceptance_masks_and_reason_counts(self):
        c, z, bc, bz = scene()
        c[10, 10] = STRONG
        c[10:12, 11:13] = WEAK
        z[10:12, 10:13] = 980
        r = self.run_seeded(c, z, bc, bz, settings={'min_object_area': 10}, diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertEqual(r.metrics['grown_pixels'], 0)
        self.assertTrue((r.diagnostics['fusion_decision'][r.diagnostics['filtered_mask']] == 6).all())
        for key in ('accepted_both_mask', 'accepted_color_only_mask', 'accepted_growth_mask'):
            self.assertFalse(r.diagnostics[key].any())
        self.assertEqual(sum(r.metrics['decision_counts'].values()), z.size)
        self.assertGreater(int(r.diagnostics['growth_step'].max()), 0)  # Pre-area provenance retained.

    def test_validation(self):
        for key, values in {'color_relaxation_ratio': [0, -1, 1.1, True, np.nan, np.inf, [1]],
                            'color_growth_max_steps': [-1, 21, 1., True, np.int64(2)]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RGBDBackgroundSegmenter(**{key: value})
        RGBDBackgroundSegmenter(color_relaxation_ratio=1, color_growth_max_steps=0)

    def test_diagnostics_units_strides_tiny_images_and_no_mutation(self):
        c, z, bc, bz = self.patch()
        saved = [a.copy() for a in (c, z, bc, bz)]
        a = self.run_seeded(c, z, bc, bz)
        d = RGBDBackgroundSegmenter(fusion_mode='color_seeded')
        b = d.detect(c[..., ::-1], z.astype(float)/1000, background_color=bc[..., ::-1],
            background_depth=bz.astype(float)/1000, color_order='BGR', depth_scale_m=1,
            registered_to_depth=True, diagnostics=True)
        np.testing.assert_array_equal(a.object_mask, b.object_mask)
        self.assertEqual(a.diagnostics, {})
        json.dumps(b.to_dict(), allow_nan=False)
        self.assertEqual(b.render_debug(c).dtype, np.uint8)
        for original, copy in zip((c, z, bc, bz), saved):
            np.testing.assert_array_equal(original, copy)
        for size in (2, 3):
            r = self.run_seeded(c[:size, :size], z[:size, :size], bc[:size, :size], bz[:size, :size],
                settings={'depth_support_radius_px': 5, 'color_growth_max_steps': 20})
            self.assertFalse(r.object_mask.any())

    def test_old_modes_ignore_new_knobs(self):
        data = self.patch()
        for mode in ('union', 'depth_required', 'evidence'):
            a = detect(*data, RGBDBackgroundSegmenter(fusion_mode=mode))
            b = detect(*data, RGBDBackgroundSegmenter(fusion_mode=mode,
                color_relaxation_ratio=.1, color_growth_max_steps=20))
            for name in ('object_mask', 'uncertain_mask', 'observed_mask', 'suppressed_mask', 'deferred_mask'):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


if __name__ == '__main__':
    unittest.main()
