"""Synthetic geometric/negative controls; not physical picking validation."""
import json
import unittest
from dataclasses import replace
from copy import deepcopy
import numpy as np
from depth_image_lens import (OverheadPickCandidateDetector, OuterContourDetector,
                              RGBDBackgroundSegmenter)


def scene():
    mask = np.zeros((80, 110), bool)
    mask[8:72, 8:85] = True
    bg = np.full(mask.shape, 1000, np.uint16)
    z = bg.copy(); z[mask] = 995
    outline = OuterContourDetector().detect(mask)
    return z, bg, outline


def detect(z, bg, outline, detector=None, **kwargs):
    return (detector or OverheadPickCandidateDetector()).detect(
        z, outline, background_depth=bg, depth_scale_m=.001,
        registered_to_mask=True, **kwargs)


class PickTests(unittest.TestCase):
    def test_supported_raised_patch_and_representative(self):
        z, bg, outline = scene()
        z[25:48, 30:55] = 950
        r = detect(z, bg, outline, diagnostics=True)
        self.assertEqual(r.state, 'candidate')
        x, y = r.point_xy
        self.assertTrue(30 <= x < 55 and 25 <= y < 48)
        self.assertTrue(r.candidate_mask[y, x])
        self.assertEqual(len(r.candidates), 1)
        item = r.candidates[0]
        self.assertGreaterEqual(item['local_support_fraction'], .8)
        self.assertGreater(item['clearance_px'], 3)
        self.assertGreater(item['region_clearance_px'], 2)
        self.assertAlmostEqual(item['local_separation_m'], .05)
        self.assertEqual(r.grasp_suitability, 'unverified')
        self.assertFalse((r.candidate_mask & ~outline.object_mask).any())
        json.dumps(r.to_dict(), allow_nan=False)
        self.assertEqual(r.render_debug(depth=z).dtype, np.uint8)

    def test_isolated_nearest_spike_does_not_win(self):
        z, bg, outline = scene()
        z[25:48, 30:55] = 950
        z[55, 65] = 200
        r = detect(z, bg, outline)
        self.assertFalse(r.candidate_mask[55, 65])
        self.assertTrue(r.unknown_mask[55, 65])
        self.assertNotEqual(r.point_xy, [65, 55])
        z[25:48, 30:55] = 995
        r = detect(z, bg, outline)
        self.assertFalse(r.candidate_mask.any())
        self.assertIsNone(r.point_xy)

    def test_background_spike_does_not_fabricate_elevation(self):
        z, bg, outline = scene()
        bg[30, 35] = 2000
        r = detect(z, bg, outline)
        self.assertFalse(r.candidate_mask.any())
        self.assertTrue(r.unknown_mask[30, 35])

    def test_boundary_mixed_depth_rejected(self):
        z, bg, outline = scene()
        z[8:11, 8:85] = 100
        r = detect(z, bg, outline)
        self.assertFalse(r.candidate_mask.any())
        self.assertFalse(r.supported_mask[8:11].any())
        self.assertTrue(r.excluded_mask[8:11, 8:85].all())

    def test_hole_nested_island_and_other_contour_are_excluded(self):
        mask = np.zeros((100, 140), bool)
        mask[5:90, 5:100] = True
        mask[30:65, 30:75] = False
        mask[40:55, 43:60] = True  # Separate object in hole.
        mask[10:50, 110:135] = True
        bg = np.full(mask.shape, 1000, np.uint16)
        z = bg.copy(); z[mask] = 950
        z[40:55, 43:60] = 100
        z[10:50, 110:135] = 200
        outline = OuterContourDetector().detect(mask)
        r = detect(z, bg, outline)
        self.assertEqual(r.contour_id, outline.largest_contour()['id'])
        self.assertFalse(r.object_mask[30:65, 30:75].any())
        self.assertFalse(r.candidate_mask[:, 110:].any())
        x, y = r.point_xy
        self.assertTrue(r.object_mask[y, x])
        self.assertGreater(z[y, x], 200)
        for contour in outline.contours:
            part = outline.component_mask(contour['id'])
            explicit = detect(z, bg, outline, contour_id=contour['id'])
            np.testing.assert_array_equal(explicit.object_mask, part)
            self.assertFalse((explicit.candidate_mask & ~part).any())

    def test_nonconvex_shape_point_is_inside_and_away_from_hole(self):
        z, bg, _ = scene()
        mask = np.zeros(z.shape, bool)
        mask[10:65, 10:30] = True; mask[45:65, 10:85] = True
        z[mask] = 940
        outline = OuterContourDetector().detect(mask)
        r = detect(z, bg, outline)
        self.assertEqual(r.state, 'candidate')
        x, y = r.point_xy
        self.assertTrue(mask[y, x])
        self.assertGreater(r.candidates[0]['clearance_px'], 3)

    def test_missing_current_reference_and_declared_masks(self):
        for target in ('current', 'background'):
            for bad in (0., np.nan, np.inf, -1., 65535.):
                z, bg, outline = scene(); z, bg = z.astype(float), bg.astype(float)
                z[25:48, 30:55] = 950
                (z if target == 'current' else bg)[30:40, 35:45] = bad
                r = detect(z, bg, outline, invalid_values=(65535,))
                self.assertFalse(r.candidate_mask[30:40, 35:45].any())
                self.assertTrue(r.unknown_mask[30:40, 35:45].all())
                if r.point_xy is not None:
                    self.assertGreater(r.candidates[0]['clearance_px'], 3)
        z, bg, outline = scene(); z[25:48, 30:55] = 950
        mask = np.ones(z.shape, bool); mask[25:48, 30:55] = False
        for name in ('depth_valid_mask', 'background_valid_mask'):
            r = detect(z, bg, outline, **{name: mask})
            self.assertFalse(r.candidate_mask.any())
            self.assertTrue(r.unknown_mask[25:48, 30:55].all())

    def test_segmentation_uncertainty_excluded_and_not_lost(self):
        z, bg, outline = scene(); z[25:48, 30:55] = 950
        uncertain = np.zeros(z.shape, bool); uncertain[25:48, 30:55] = True
        r = detect(z, bg, outline, segmentation_uncertain_mask=uncertain)
        self.assertFalse(r.candidate_mask.any())
        self.assertTrue(r.unknown_mask[uncertain].all())
        self.assertTrue(r.excluded_mask[uncertain].all())

    def test_empty_invalid_and_no_elevation(self):
        z, bg, outline = scene()
        r = detect(z, bg, outline)
        self.assertEqual(r.state, 'no_candidate')
        r = detect(z * 0, bg, outline)
        self.assertEqual(r.state, 'unknown')
        self.assertIsNone(r.point_xy)
        empty = OuterContourDetector().detect(np.zeros(z.shape, bool))
        r = detect(z, bg, empty)
        self.assertEqual(r.state, 'no_object')
        self.assertIsNone(r.contour_id)
        with self.assertRaises(ValueError): detect(z, bg, empty, contour_id=0)

    def test_near_camera_without_reference_elevation_is_not_candidate(self):
        z, bg, outline = scene()
        z[outline.object_mask] = 400; bg[:] = 405
        r = detect(z, bg, outline)
        self.assertEqual(r.state, 'no_candidate')
        self.assertFalse(r.candidate_mask.any())

    def test_absolute_depth_ranks_not_maximum_reference_difference(self):
        z, bg, outline = scene()
        z[20:40, 20:40] = 800; bg[20:40, 20:40] = 850  # nearer, less elevation
        z[45:65, 55:75] = 900; bg[45:65, 55:75] = 1100
        r = detect(z, bg, outline)
        self.assertEqual(len(r.candidates), 2)
        x, y = r.point_xy
        self.assertTrue(20 <= x < 40 and 20 <= y < 40)
        self.assertLess(r.candidates[0]['local_separation_m'], r.candidates[1]['local_separation_m'])

    def test_area_and_width_constraints(self):
        z, bg, outline = scene(); z[25:48, 30:55] = 950
        for d in (OverheadPickCandidateDetector(min_region_area=1000),
                  OverheadPickCandidateDetector(min_region_radius_px=30)):
            r = detect(z, bg, outline, d)
            self.assertFalse(r.candidate_mask.any())
            self.assertTrue(r.metrics['rejected_regions'])
        mask = np.zeros(z.shape, bool); mask[20:65, 20:23] = True
        r = detect(z, bg, OuterContourDetector().detect(mask))
        self.assertEqual(r.state, 'unknown')
        self.assertIn('no_trusted_interior_with_clearance', r.reasons)

    def test_rough_checkerboard_and_broad_flat_sheet_counterexamples(self):
        z, bg, outline = scene()
        yy, xx = np.indices(z.shape)
        z[outline.object_mask] = np.where((yy + xx)[outline.object_mask] % 2, 930, 970)
        r = detect(z, bg, outline)
        self.assertFalse(r.candidate_mask.any())
        self.assertEqual(r.state, 'unknown')
        z[outline.object_mask] = 950
        r = detect(z, bg, outline)
        self.assertEqual(r.state, 'candidate')  # Geometrically valid, no pinch-access claim.
        self.assertEqual(r.grasp_suitability, 'unverified')

    def test_frame_and_observation_cut_have_clearance(self):
        z, bg, _ = scene(); z[:] = 950
        mask = np.ones(z.shape, bool)
        observed = mask.copy(); observed[:, 50:55] = False
        outline = OuterContourDetector().detect(mask, observed_mask=observed)
        for c in outline.contours:
            r = detect(z, bg, outline, contour_id=c['id'])
            self.assertEqual(r.state, 'candidate')
            self.assertFalse(r.candidate_mask[:, 47:58].any())
            self.assertFalse(r.candidate_mask[:3].any())
            self.assertFalse(r.candidate_mask[-3:].any())

    def test_support_tuning_rejects_inconsistent_center(self):
        z, bg, outline = scene(); z[25:48, 30:55] = 950
        z[35, 40] = 900
        r = detect(z, bg, outline)
        self.assertFalse(r.supported_mask[35, 40])
        self.assertFalse(r.candidate_mask[35, 40])
        self.assertTrue(r.unknown_mask[35, 40])

    def test_metre_scale_strides_and_no_mutation(self):
        z, bg, outline = scene(); z[25:48, 30:55] = 950
        before = (z.copy(), bg.copy(), outline.object_mask.copy())
        r = detect(z, bg, outline)
        zm, bm = z.astype(float) / 1000, bg.astype(float) / 1000
        other = OverheadPickCandidateDetector().detect(zm, outline, background_depth=bm,
                       depth_scale_m=1., registered_to_mask=True)
        np.testing.assert_array_equal(r.candidate_mask, other.candidate_mask)
        self.assertEqual(r.point_xy, other.point_xy)
        for a, b in zip((z, bg, outline.object_mask), before): np.testing.assert_array_equal(a, b)
        flipped = OuterContourDetector().detect(outline.object_mask[:, ::-1])
        r = detect(z[:, ::-1], bg[:, ::-1], flipped)
        self.assertEqual(r.state, 'candidate')

    def test_rgbd_end_to_end(self):
        z, bg, outline = scene()
        z[25:48, 30:55] = 950
        bc = np.full((*z.shape, 3), 140, np.uint8); color = bc.copy()
        color[outline.object_mask] = (25, 180, 80)
        seg = RGBDBackgroundSegmenter().detect(color, z, background_color=bc, background_depth=bg,
                        color_order='RGB', registered_to_depth=True, depth_scale_m=.001)
        fresh = OuterContourDetector().detect(seg.object_mask, observed_mask=seg.observed_mask)
        r = detect(z, bg, fresh, segmentation_uncertain_mask=seg.uncertain_mask,
                   depth_valid_mask=seg.depth_valid_mask)
        self.assertEqual(r.state, 'candidate')
        x, y = r.point_xy
        self.assertTrue(30 <= x < 55 and 25 <= y < 48)

    def test_input_parameter_validation(self):
        for kwargs in ({'local_radius_px': 0}, {'local_radius_px': 11}, {'local_radius_px': True},
                       {'min_region_area': 0}, {'min_support_fraction': .5}, {'min_support_fraction': 1.1},
                       {'min_elevation_m': 0}, {'local_depth_tolerance_m': 0}, {'boundary_guard_px': -1},
                       {'nearest_depth_band_m': np.nan}, {'reference_error_m': np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OverheadPickCandidateDetector(**kwargs)
        z, bg, outline = scene()
        base = dict(background_depth=bg, depth_scale_m=.001, registered_to_mask=True)
        for key, value in [('registered_to_mask', False), ('depth_scale_m', 0), ('background_depth', bg[:-1]),
                           ('depth_valid_mask', np.ones(z.shape, np.uint8)), ('contour_id', True),
                           ('segmentation_uncertain_mask', np.zeros((2, 2), bool)), ('diagnostics', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                OverheadPickCandidateDetector().detect(z, outline, **dict(base, **{key: value}))

    def test_installed_component_method_keeps_validation_and_id_lookup(self):
        z, bg, outline = scene()
        records = deepcopy(outline.contours); records[0]['id'] = 41
        altered = replace(outline, contours=records)
        np.testing.assert_array_equal(altered.component_mask(41), outline.object_mask)
        records[0]['points_xy'][0] = [-1, 10]
        with self.assertRaises(ValueError): altered.component_mask(41)

    def test_repeatable_small_depth_noise_with_a_large_outlier(self):
        rng = np.random.default_rng(271)
        for _ in range(5):
            z, bg, outline = scene()
            z = z.astype(float); bg = bg.astype(float)
            z[25:48, 30:55] = 950
            z += rng.normal(0, .4, z.shape)
            bg += rng.normal(0, .4, bg.shape)
            z[55, 65] = 200
            result = detect(z, bg, outline)
            self.assertEqual(result.state, 'candidate')
            x, y = result.point_xy
            self.assertTrue(30 <= x < 55 and 25 <= y < 48)
            self.assertFalse(result.candidate_mask[55, 65])

    def test_reference_error_changes_eligibility(self):
        z, bg, outline = scene(); z[25:48, 30:55] = 985
        accepted = detect(z, bg, outline)
        rejected = detect(z, bg, outline, OverheadPickCandidateDetector(reference_error_m=.01))
        self.assertEqual(accepted.state, 'candidate')
        self.assertFalse(rejected.candidate_mask.any())

    def test_missing_support_fraction_is_measured_against_whole_window(self):
        z, bg, outline = scene(); z[outline.object_mask] = 950
        valid = np.ones(z.shape, bool); valid[35, 40] = False
        # Guard disabled explicitly so the local support rule can be isolated.
        loose = detect(z, bg, outline, OverheadPickCandidateDetector(boundary_guard_px=0), depth_valid_mask=valid)
        strict = detect(z, bg, outline, OverheadPickCandidateDetector(boundary_guard_px=0, min_support_fraction=1.), depth_valid_mask=valid)
        self.assertTrue(loose.supported_mask[35, 41])
        self.assertFalse(strict.supported_mask[35, 41])
        self.assertFalse(loose.candidate_mask[35, 40])

    def test_metric_overflow_is_unknown(self):
        z, bg, outline = scene()
        huge = np.full(z.shape, 1e308)
        result = OverheadPickCandidateDetector().detect(
            huge, outline, background_depth=huge, depth_scale_m=10., registered_to_mask=True)
        self.assertEqual(result.state, 'unknown')
        self.assertIsNone(result.point_xy)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_multiple_offset_regions_keep_global_coordinates_and_clearance(self):
        z, bg, outline = scene()
        boxes = [(15, 28, 15, 28), (40, 57, 55, 74)]
        z[15:28, 15:28] = 950
        z[40:57, 55:74] = 930
        result = detect(z, bg, outline)
        self.assertEqual(len(result.candidates), 2)
        for candidate, (y0, y1, x0, x1) in zip(result.candidates, boxes[::-1]):
            x, y = candidate['point_xy']
            self.assertTrue(x0 <= x < x1 and y0 <= y < y1)
            self.assertTrue(result.candidate_mask[y, x])
            self.assertGreater(candidate['region_clearance_px'], 2)
        self.assertEqual(result.point_xy, result.candidates[0]['point_xy'])


if __name__ == '__main__':
    unittest.main()
