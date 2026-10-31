"""Synthetic evidence-contract regressions, not real-image accuracy measurements."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector, OverheadBoundaryDetector


def scene():
    color = np.full((40, 60, 3), 140, np.uint8)
    depth = np.full((40, 60), 1000, np.uint16)
    return color.copy(), depth.copy(), color, depth


def detect(c, z, bc, bz, detector=None, **kwargs):
    return (detector or RGBDBackgroundSegmenter()).detect(
        c, z, background_color=bc, background_depth=bz, color_order='RGB',
        registered_to_depth=True, depth_scale_m=.001, **kwargs)


class RGBDBackgroundTests(unittest.TestCase):
    def test_same_color_nonconvex_holes_and_components(self):
        c, z, bc, bz = scene()
        expected = np.zeros(z.shape, bool)
        expected[3:28, 3:27] = True
        expected[3:12, 17:27] = False  # Concavity.
        expected[15:22, 8:15] = False  # Real background hole.
        expected[28:35, 42:51] = True
        z[expected] = 970
        r = detect(c, z, bc, bz)
        np.testing.assert_array_equal(r.object_mask, expected)
        self.assertFalse(r.color_foreground_mask.any())
        outline = OuterContourDetector().detect(r.object_mask, observed_mask=r.observed_mask)
        self.assertEqual(len(outline.contours), 2)
        self.assertEqual(len(outline.holes), 1)
        boundary = OverheadBoundaryDetector(min_object_area=1).detect(c, object_mask=r.object_mask, color_order='RGB')
        np.testing.assert_array_equal(boundary.object_mask, expected)

    def test_color_only_thin_object_is_uncertain(self):
        c, z, bc, bz = scene()
        c[10:25, 10:30] = (30, 150, 210)
        z[10:25, 10:30] = 999
        r = detect(c, z, bc, bz)
        self.assertTrue(r.object_mask[10:25, 10:30].all())
        self.assertTrue(r.uncertain_mask[r.object_mask].all())
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='depth_required'))
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.uncertain_mask[10:25, 10:30].all())

    def test_mild_shadow_suppressed_but_dark_shadow_ambiguous(self):
        c, z, bc, bz = scene()
        c[5:15, 5:20] = 120
        self.assertFalse(detect(c, z, bc, bz).object_mask.any())
        c[5:15, 5:20] = 40
        r = detect(c, z, bc, bz)
        self.assertTrue(r.object_mask[5:15, 5:20].all())
        self.assertTrue(r.uncertain_mask[r.object_mask].all())
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='depth_required'))
        self.assertFalse(r.object_mask.any())

    def test_farther_change_veto_and_unsigned_subtraction(self):
        c, z, bc, bz = scene()
        c[5:15, 5:20] = (10, 220, 100)
        z[5:15, 5:20] = 1100
        r = detect(c, z, bc, bz, diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.conflict_mask[5:15, 5:20].all())
        np.testing.assert_allclose(r.diagnostics['depth_difference_m'][5:15, 5:20], -.1)

    def test_missing_depth_policies_and_unknown_contour_domain(self):
        for missing_reference in (False, True):
            for value in (0., -1., np.nan, np.inf, 65535.):
                c, z, bc, bz = scene()
                z, bz = z.astype(float), bz.astype(float)
                c[5:15, 5:20] = (10, 220, 100)
                (bz if missing_reference else z)[5:15, 5:20] = value
                r = detect(c, z, bc, bz, invalid_values=(65535,), diagnostics=True)
                self.assertTrue(r.object_mask[5:15, 5:20].all())
                self.assertTrue(r.uncertain_mask[5:15, 5:20].all())
                self.assertTrue(np.isnan(r.diagnostics['depth_difference_m'][5:15, 5:20]).all())
                for d in (RGBDBackgroundSegmenter(missing_depth_policy='unknown'),
                          RGBDBackgroundSegmenter(fusion_mode='depth_required')):
                    r = detect(c, z, bc, bz, d, invalid_values=(65535,))
                    self.assertFalse(r.object_mask.any())
                    self.assertFalse(r.observed_mask[5:15, 5:20].any())

    def test_separate_validity_and_roi(self):
        c, z, bc, bz = scene()
        z[10:25, 10:30] = 970
        c[10:25, 10:30] = (10, 220, 100)
        valid = np.ones(z.shape, bool)
        valid[10:25, 10:30] = False
        for name in ('color_valid_mask', 'background_color_valid_mask'):
            r = detect(c, z, bc, bz, **{name: valid})
            self.assertTrue(r.object_mask[10:25, 10:30].all())
            self.assertFalse(r.color_foreground_mask.any())
            self.assertTrue(r.uncertain_mask[r.object_mask].all())
        for name in ('depth_valid_mask', 'background_depth_valid_mask'):
            r = detect(c, z, bc, bz, **{name: valid})
            self.assertFalse(r.depth_valid_mask[10:25, 10:30].any())
            self.assertTrue(r.object_mask[10:25, 10:30].all())
        r = detect(c, z, bc, bz, color_valid_mask=valid, depth_valid_mask=valid)
        self.assertFalse(r.object_mask.any())
        self.assertFalse(r.observed_mask[10:25, 10:30].any())
        r = detect(c, z, bc, bz, work_roi=valid)
        self.assertFalse(r.object_mask.any())
        self.assertFalse(r.uncertain_mask[~valid].any())
        self.assertFalse(r.observed_mask[~valid].any())

    def test_illumination_compensation_on_declared_background(self):
        c, z, bc, bz = scene()
        c[:] = 155
        check = np.zeros(z.shape, bool)
        check[:5] = True
        d = RGBDBackgroundSegmenter(background_luma_tolerance=5, background_compensate=True)
        r = detect(c, z, bc, bz, d, background_check_mask=check)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.stages['color_reference']['compensation_applied'])
        with self.assertRaises(ValueError):
            detect(c, z, bc, bz, d)

    def test_rejected_color_check_retains_depth_without_border_fallback(self):
        c, z, bc, bz = scene()
        c[:] = (20, 220, 100)
        z[10:20, 10:20] = 970
        check = np.zeros(z.shape, bool)
        check[:5] = True
        r = detect(c, z, bc, bz, background_check_mask=check)
        self.assertEqual(r.stages['color_reference']['status'], 'rejected')
        self.assertNotIn('fallback', r.stages['color_reference'])
        self.assertEqual(r.object_mask.sum(), 100)
        self.assertFalse(r.color_foreground_mask.any())

    def test_noise_threshold_and_margin(self):
        c, z, bc, bz = scene()
        z[0, :6] = (999, 998, 997, 996, 995, 994)
        d = RGBDBackgroundSegmenter(depth_threshold_m=.005, depth_margin_m=.0015)
        r = detect(c, z, bc, bz, d)
        self.assertFalse(r.object_mask[0, :4].any())
        self.assertTrue(r.object_mask[0, 5])
        self.assertTrue(r.uncertain_mask[0, 3:6].all())

    def test_exact_threshold_is_exclusive(self):
        c, z, bc, bz = scene()
        z = z.astype(float) / 1000
        bz = bz.astype(float) / 1000
        z[5, 5] = .875
        r = RGBDBackgroundSegmenter(depth_threshold_m=.125, depth_margin_m=0).detect(
            c, z, background_color=bc, background_depth=bz, color_order='RGB',
            depth_scale_m=1, registered_to_depth=True)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.uncertain_mask[5, 5])

    def test_area_filter_preserves_holes_and_other_components(self):
        c, z, bc, bz = scene()
        z[5:15, 5:15] = 970
        z[8:12, 8:12] = 1000
        z[25, 30] = 970
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(min_object_area=5), diagnostics=True)
        self.assertEqual(r.object_mask.sum(), 84)
        self.assertTrue(r.uncertain_mask[25, 30])
        self.assertTrue(r.diagnostics['filtered_mask'][25, 30])
        self.assertFalse(r.object_mask[8:12, 8:12].any())

    def test_area_filter_removes_many_speckles_without_removing_large_object(self):
        c, z, bc, bz = scene()
        z[::3, ::3] = 970
        z[15:28, 24:40] = 970
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(min_object_area=5), diagnostics=True)
        self.assertTrue(r.object_mask[15:28, 24:40].all())
        self.assertFalse(r.object_mask[:10].any())
        self.assertTrue(r.diagnostics['filtered_mask'][0, 0])
        self.assertFalse(r.diagnostics['filtered_mask'][20, 30])

    def test_no_change_is_not_object_absence(self):
        r = detect(*scene())
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.observed_mask.all())
        self.assertIn('no_change_does_not_prove_no_object', r.stages['fusion']['warnings'])
        self.assertEqual(r.diagnostics, {})
        json.dumps(r.to_dict(), allow_nan=False)

    def test_unobserved_and_empty_domains_are_not_no_change(self):
        c, z, bc, bz = scene()
        absent = np.zeros(z.shape, bool)
        r = detect(c, z, bc, bz, color_valid_mask=absent, depth_valid_mask=absent)
        self.assertEqual(r.stages['fusion']['status'], 'insufficient')
        self.assertTrue(r.uncertain_mask.all())
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='depth_required'), depth_valid_mask=absent)
        self.assertEqual(r.stages['fusion']['status'], 'insufficient')
        r = detect(c, z, bc, bz, work_roi=absent)
        self.assertEqual(r.stages['fusion']['status'], 'empty_domain')
        self.assertFalse(r.observed_mask.any())

    def test_rejected_or_unused_color_is_not_reported_as_applied(self):
        c, z, bc, bz = scene()
        r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(fusion_mode='depth_required'))
        self.assertFalse(r.stages['color_reference']['difference_model']['applied_to_selection'])
        absent = np.zeros(z.shape, bool)
        for kwargs in ({'background_compensate': True}, {}):
            with self.assertRaises(ValueError):
                detect(c, z, bc, bz, RGBDBackgroundSegmenter(**kwargs),
                       color_valid_mask=absent, background_check_mask=~absent)

    def test_units_order_strides_no_input_mutation_and_debug(self):
        c, z, bc, bz = scene()
        c[5:20, 5:20] = (10, 220, 100)
        z[5:20, 5:20] = 970
        saved = [x.copy() for x in (c, z, bc, bz)]
        r = detect(c, z, bc, bz, diagnostics=True)
        other = RGBDBackgroundSegmenter().detect(
            c[..., ::-1], z.astype(float) / 1000, background_color=bc[..., ::-1],
            background_depth=bz.astype(float) / 1000, color_order='BGR',
            registered_to_depth=True, depth_scale_m=1, diagnostics=True)
        np.testing.assert_array_equal(r.object_mask, other.object_mask)
        self.assertEqual(r.render_debug(c).dtype, np.uint8)
        for actual, original in zip((c, z, bc, bz), saved):
            np.testing.assert_array_equal(actual, original)
        for key in ('background_chroma_distance', 'background_luma_distance', 'background_combined_score', 'depth_difference_m'):
            self.assertIn(key, r.diagnostics)

    def test_clipping_is_uncertain(self):
        c, z, bc, bz = scene()
        c[5:20, 5:20] = 255
        r = detect(c, z, bc, bz)
        self.assertTrue(r.uncertain_mask[5:20, 5:20].all())

    def test_constructor_validation(self):
        for key in ('background_chroma_tolerance', 'background_luma_tolerance', 'depth_threshold_m',
                    'background_check_p95', 'background_max_offset'):
            for value in (0, -1, np.inf, np.nan, True, [1]):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    RGBDBackgroundSegmenter(**{key: value})
        for kwargs in ({'fusion_mode': 'and'}, {'missing_depth_policy': 'fill'},
                       {'min_object_area': True}, {'min_object_area': 0},
                       {'background_score_margin': 2}, {'depth_margin_m': .005},
                       {'background_compensate': 1}):
            with self.assertRaises(ValueError):
                RGBDBackgroundSegmenter(**kwargs)

    def test_input_contract_validation(self):
        c, z, bc, bz = scene()
        base = dict(background_color=bc, background_depth=bz, color_order='RGB', registered_to_depth=True, depth_scale_m=.001)
        for key, value in [('registered_to_depth', False), ('color_order', 'guess'),
                           ('background_color', bc.astype(float)), ('background_depth', bz[:-1]),
                           ('depth_scale_m', 0), ('depth_scale_m', np.nan), ('diagnostics', 1),
                           ('work_roi', np.ones(z.shape, np.uint8)), ('depth_valid_mask', np.ones((2, 2), bool)),
                           ('background_check_mask', np.zeros(z.shape, bool))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                RGBDBackgroundSegmenter().detect(c, z, **dict(base, **{key: value}))
        for bad in (np.zeros(z.shape, bool), np.zeros(z.shape, complex), z[:-1]):
            with self.assertRaises(ValueError):
                RGBDBackgroundSegmenter().detect(c, bad, **base)

    def test_extreme_depth_is_missing_not_foreground(self):
        c, z, bc, bz = scene()
        z = np.full(z.shape, 1e308)
        bz = z.copy()
        r = RGBDBackgroundSegmenter().detect(c, z, background_color=bc, background_depth=bz,
                color_order='RGB', registered_to_depth=True, depth_scale_m=10, diagnostics=True)
        self.assertFalse(r.depth_valid_mask.any())
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.uncertain_mask.all())
        self.assertTrue(np.isnan(r.diagnostics['depth_difference_m']).all())


if __name__ == '__main__':
    unittest.main()
