"""Geometric repair contracts and explicit counterexamples, not field accuracy."""
import json
import subprocess
import sys
import unittest
import numpy as np
from depth_image_lens import cleanup_mask, MaskCleanupResult
from depth_image_lens.mask_cleanup import _components8
from examples.example_mask_cleanup import _component_count, run_experiments, rgbd_pipeline_example


def rectangle():
    mask = np.zeros((32, 40), bool)
    mask[4:28, 4:36] = True
    return mask


class MaskCleanupTests(unittest.TestCase):
    def test_defaults_are_independent_identity_snapshots(self):
        raw = rectangle()
        raw[12, 12] = False
        result = cleanup_mask(raw)
        self.assertIsInstance(result, MaskCleanupResult)
        np.testing.assert_array_equal(result.object_mask, raw)
        self.assertFalse(result.changed_mask.any())
        self.assertIsNone(result.metrics['enclosed_holes'])
        result.object_mask[:] = False
        np.testing.assert_array_equal(result.original_mask, raw)
        self.assertTrue(raw.any())
        json.dumps(result.to_dict(), allow_nan=False)

    def test_area_limit_is_inclusive_and_independent(self):
        raw = rectangle()
        raw[10:12, 10:12] = False  # Four pixels.
        raw[18, 20:25] = False  # Five pixels.
        expected = raw.copy()
        expected[10:12, 10:12] = True
        result = cleanup_mask(raw, max_hole_area=4)
        np.testing.assert_array_equal(result.object_mask, expected)
        self.assertEqual(result.metrics['filled_holes'], 1)
        self.assertEqual(result.metrics['filled_hole_pixels'], 4)
        self.assertFalse(result.smoothed_added_mask.any())
        self.assertFalse(result.smoothed_removed_mask.any())

    def test_frame_open_and_diagonal_leaks_are_not_holes(self):
        for diagonal in (False, True):
            raw = rectangle()
            if diagonal:
                raw[np.arange(4, 15), np.arange(4, 15)] = False
            else:
                raw[:15, 14] = False
            result = cleanup_mask(raw, max_hole_area=1000)
            np.testing.assert_array_equal(result.object_mask, raw)

    def test_hole_and_its_rim_must_be_observed_and_unprotected(self):
        raw = rectangle()
        raw[14:17, 14:17] = False
        for key in ('observed_mask', 'protected_mask'):
            for location in ((15, 15), (13, 15), (13, 13)):
                values = np.ones(raw.shape, bool) if key == 'observed_mask' else np.zeros(raw.shape, bool)
                values[location] = key != 'observed_mask'
                result = cleanup_mask(raw, max_hole_area=9, **{key: values})
                np.testing.assert_array_equal(result.object_mask, raw)

    def test_hole_with_separate_island_does_not_merge(self):
        raw = rectangle()
        raw[9:22, 10:25] = False
        raw[14:17, 16:19] = True
        result = cleanup_mask(raw, max_hole_area=1000)
        np.testing.assert_array_equal(result.object_mask, raw)
        self.assertEqual(_component_count(result.object_mask), 2)

    def test_hole_filling_has_unavoidable_true_hole_counterexample(self):
        raw = rectangle()
        raw[14, 14] = False
        result = cleanup_mask(raw, max_hole_area=1)
        self.assertTrue(result.object_mask[14, 14])
        self.assertIn('small_real_holes_and_segmentation_holes_are_indistinguishable', result.warnings)

    def test_smoothing_does_not_fill_or_reshape_retained_holes(self):
        raw = rectangle()
        raw[12:18, 13:20] = False
        raw[11, 15] = False
        for radius in (1, 2, 5):
            result = cleanup_mask(raw, smoothing_radius_px=radius)
            np.testing.assert_array_equal(result.object_mask[10:19, 12:21], raw[10:19, 12:21])
            self.assertFalse(result.filled_hole_mask.any())

    def test_separate_objects_are_not_joined(self):
        raw = rectangle()
        raw[:, 19] = False
        for radius in (1, 2, 5):
            result = cleanup_mask(raw, smoothing_radius_px=radius)
            self.assertEqual(_component_count(result.object_mask), 2)

    def test_thin_neck_and_isolated_component_survive(self):
        raw = np.zeros((40, 60), bool)
        raw[10:30, 5:22] = True
        raw[10:30, 38:55] = True
        raw[20, 22:38] = True
        raw[4, 4] = True
        for radius in (1, 2, 5):
            result = cleanup_mask(raw, smoothing_radius_px=radius)
            self.assertEqual(_component_count(result.object_mask), 2)
            self.assertTrue(result.object_mask[4, 4])
            self.assertTrue(result.object_mask[20, 30])

    def test_smoothing_can_round_real_corners_and_shorten_appendages(self):
        raw = rectangle()
        raw[15, 36:39] = True
        result = cleanup_mask(raw, smoothing_radius_px=1)
        self.assertFalse(result.object_mask[4, 4])
        self.assertGreater(result.metrics['removed_pixels'], 0)
        self.assertFalse(result.object_mask[15, 38])
        protected = raw.copy()
        safe = cleanup_mask(raw, smoothing_radius_px=1, protected_mask=protected)
        np.testing.assert_array_equal(safe.object_mask, raw)

    def test_cropped_frame_and_unknown_halo_are_fixed(self):
        raw = rectangle()
        raw[:5, 4:36] = True
        observed = np.ones(raw.shape, bool)
        observed[12, 4] = False
        protected = np.zeros(raw.shape, bool)
        protected[20, 35] = True
        result = cleanup_mask(raw, max_hole_area=9, smoothing_radius_px=2,
                              observed_mask=observed, protected_mask=protected)
        np.testing.assert_array_equal(result.object_mask[:2], raw[:2])
        np.testing.assert_array_equal(result.object_mask[10:15, 2:7], raw[10:15, 2:7])
        np.testing.assert_array_equal(result.object_mask[18:23, 33:38], raw[18:23, 33:38])
        np.testing.assert_array_equal(result.observed_mask, observed)
        np.testing.assert_array_equal(result.protected_mask, protected)

    def test_empty_full_and_tiny_arrays(self):
        for shape in ((1, 1), (1, 6), (2, 2), (4, 3), (32, 40)):
            for value in (False, True):
                raw = np.full(shape, value, bool)
                result = cleanup_mask(raw, max_hole_area=10, smoothing_radius_px=5)
                np.testing.assert_array_equal(result.object_mask, raw)

    def test_noncontiguous_inputs_nonmutation_and_determinism(self):
        raw = rectangle()[:, ::-1]
        raw[12, 12] = False
        observed = np.ones((32, 80), bool)[:, ::2]
        protected = np.zeros((32, 80), bool)[:, ::2]
        before = [a.copy() for a in (raw, observed, protected)]
        a = cleanup_mask(raw, max_hole_area=4, smoothing_radius_px=1,
                         observed_mask=observed, protected_mask=protected)
        b = cleanup_mask(raw.copy(), max_hole_area=4, smoothing_radius_px=1,
                         observed_mask=observed.copy(), protected_mask=protected.copy())
        np.testing.assert_array_equal(a.object_mask, b.object_mask)
        for old, current in zip(before, (raw, observed, protected)):
            np.testing.assert_array_equal(old, current)
        np.testing.assert_array_equal(a.added_mask, a.object_mask & ~a.original_mask)
        np.testing.assert_array_equal(a.removed_mask, a.original_mask & ~a.object_mask)
        np.testing.assert_array_equal(a.added_mask, a.filled_hole_mask | a.smoothed_added_mask)
        np.testing.assert_array_equal(a.removed_mask, a.smoothed_removed_mask)

    def test_strict_input_and_settings_validation(self):
        raw = rectangle()
        for bad in (raw.astype(np.uint8), np.zeros((0, 3), bool), np.zeros((2, 2, 2), bool), [True]):
            with self.assertRaises(ValueError):
                cleanup_mask(bad)
        for key in ('max_hole_area', 'smoothing_radius_px'):
            for bad in (True, np.int64(1), -1, 1.0, np.nan, '1', None):
                with self.assertRaises(ValueError):
                    cleanup_mask(raw, **{key: bad})
        with self.assertRaises(ValueError):
            cleanup_mask(raw, smoothing_radius_px=6)
        for key in ('observed_mask', 'protected_mask'):
            for bad in (raw.astype(np.uint8), raw[:-1]):
                with self.assertRaises(ValueError):
                    cleanup_mask(raw, **{key: bad})

    def test_numpy_only_import_and_execution(self):
        code = '''
import sys
sys.modules['cv2'] = None
from depth_image_lens import cleanup_mask
import numpy as np
m = np.ones((9, 9), bool); m[4, 4] = False
assert cleanup_mask(m, max_hole_area=1, smoothing_radius_px=1).object_mask.all()
'''
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_run_length_components_and_random_topology_against_flood_fill(self):
        random = np.random.default_rng(1204)
        for _ in range(45):
            raw = random.random((12, 14)) < random.uniform(.15, .85)
            labels, areas = _components8(raw)
            self.assertEqual(len(areas) - 1, _component_count(raw))
            self.assertEqual(int(areas[1:].sum()), int(raw.sum()))
            np.testing.assert_array_equal(labels > 0, raw)
            result = cleanup_mask(raw, smoothing_radius_px=1)
            self.assertEqual(_component_count(result.object_mask), _component_count(raw))
            self.assertEqual(_component_count(~result.object_mask), _component_count(~raw))

    def test_synthetic_metrics_include_success_and_failure(self):
        rows = {row['case']: row for row in run_experiments()}
        for case in ('small_segmentation_holes', 'jagged_exterior', 'holes_and_jagged_exterior'):
            self.assertGreater(rows[case]['cleaned_iou'], rows[case]['raw_iou'])
        self.assertGreater(rows['real_small_hole_is_indistinguishable']['false_foreground_added'], 0)
        self.assertGreater(rows['sharp_corners_can_round']['true_foreground_removed'], 0)
        self.assertEqual(rows['neighbor_objects_do_not_merge']['cleaned_components'], 2)
        self.assertEqual(rows['protected_corners_preserved']['cleaned_error_pixels'], 0)
        self.assertEqual(rows['missing_object_not_invented']['added_pixels'], 0)
        json.dumps(rows, allow_nan=False)

    def test_rgbd_to_outline_integration_is_explicit(self):
        report = rgbd_pipeline_example()
        self.assertEqual(report['cleanup']['metrics']['filled_hole_pixels'], 4)
        self.assertEqual(report['outline_summary']['component_count'], 1)
        self.assertGreater(report['combined_uncertain_pixels'], 4)
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
