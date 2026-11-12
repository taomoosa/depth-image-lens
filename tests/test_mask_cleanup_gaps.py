"""Opt-in open-gap repair, including deliberately unsafe semantic examples."""
import json
import subprocess
import sys
import unittest
import numpy as np
from depth_image_lens import cleanup_mask
from examples.example_mask_cleanup import _component_count


def notch(width=2, length=20):
    mask = np.zeros((64, 80), bool)
    mask[12:52, 12:68] = True
    mask[12:12 + length, 38:38 + width] = False
    return mask


class MaskGapTests(unittest.TestCase):
    def test_defaults_and_either_zero_disable_closing(self):
        raw = notch()
        for settings in ({}, {'closing_radius_px': 2}, {'max_gap_area': 80}):
            r = cleanup_mask(raw, **settings)
            np.testing.assert_array_equal(r.object_mask, raw)
            self.assertFalse(r.closed_gap_mask.any())

    def test_hole_area_does_not_close_exterior_channel(self):
        raw = notch()
        np.testing.assert_array_equal(cleanup_mask(raw, max_hole_area=100000).object_mask, raw)
        # Smoothing preserves an open background path even if it reshapes it.
        for radius in (1, 2, 5):
            r = cleanup_mask(raw, smoothing_radius_px=radius)
            self.assertFalse(r.object_mask[12:32, 38:40].all())

    def test_radius_width_and_inclusive_area(self):
        for width in range(1, 7):
            raw = notch(width)
            area = width * 20
            radius = (width + 1) // 2
            r = cleanup_mask(raw, closing_radius_px=radius, max_gap_area=area)
            self.assertEqual(int(r.closed_gap_mask.sum()), area)
            self.assertTrue(r.object_mask[12:32, 38:38 + width].all())
            self.assertFalse(r.removed_mask.any())
            rejected = cleanup_mask(raw, closing_radius_px=radius, max_gap_area=area - 1)
            np.testing.assert_array_equal(rejected.object_mask, raw)
            if radius > 1:
                smaller = cleanup_mask(raw, closing_radius_px=radius - 1, max_gap_area=area)
                np.testing.assert_array_equal(smaller.object_mask, raw)

    def test_separate_objects_never_merge(self):
        raw = notch()
        raw[:, 38:40] = False  # Two distinct pieces, not a notch in one piece.
        for radius in (1, 2, 5):
            r = cleanup_mask(raw, closing_radius_px=radius, max_gap_area=10000)
            np.testing.assert_array_equal(r.object_mask, raw)
            self.assertEqual(_component_count(r.object_mask), 2)

    def test_complete_candidate_rejected_near_unknown_or_protected(self):
        raw = notch()
        for key in ('observed_mask', 'protected_mask'):
            for point in ((20, 39), (20, 36)):  # In gap or two-pixel footprint.
                values = np.ones_like(raw) if key == 'observed_mask' else np.zeros_like(raw)
                values[point] = key == 'protected_mask'
                r = cleanup_mask(raw, closing_radius_px=1, max_gap_area=80, **{key: values})
                np.testing.assert_array_equal(r.object_mask, raw)

    def test_frame_contact_rejects_entire_candidate(self):
        raw = np.zeros((40, 50), bool)
        raw[:30, 5:45] = True
        raw[:20, 24:26] = False
        r = cleanup_mask(raw, closing_radius_px=1, max_gap_area=100)
        np.testing.assert_array_equal(r.object_mask, raw)

    def test_retained_enclosed_holes_and_rim_unchanged(self):
        raw = notch()
        raw[40:43, 20:23] = False
        r = cleanup_mask(raw, closing_radius_px=2, max_gap_area=100)
        self.assertEqual(int(r.closed_gap_mask.sum()), 40)
        np.testing.assert_array_equal(r.object_mask[39:44, 19:24], raw[39:44, 19:24])

    def test_bottle_neck_does_not_fill_new_pocket_in_same_call(self):
        raw = notch(width=2, length=12)
        raw[23:35, 33:47] = False
        r = cleanup_mask(raw, closing_radius_px=1, max_gap_area=100, max_hole_area=1000)
        self.assertTrue(r.closed_gap_mask.any())
        self.assertFalse(r.object_mask[29, 40])
        self.assertFalse(r.filled_hole_mask.any())

    def test_real_notch_is_indistinguishable(self):
        raw = notch()
        r = cleanup_mask(raw, closing_radius_px=1, max_gap_area=40)
        self.assertEqual(int(r.added_mask.sum()), 40)  # Wrong if raw notch was real.
        self.assertIn('closing_can_seal_real_notches_and_create_enclosed_pockets', r.warnings)

    def test_provenance_and_smoothing_composition(self):
        raw = notch()
        raw[40:42, 20:22] = False
        before = raw.copy()
        r = cleanup_mask(raw[:, ::-1], closing_radius_px=1, max_gap_area=40,
                         max_hole_area=4, smoothing_radius_px=1)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_array_equal(r.added_mask,
            r.closed_gap_mask | r.filled_hole_mask | r.smoothed_added_mask)
        np.testing.assert_array_equal(r.removed_mask, r.smoothed_removed_mask)
        self.assertEqual(int(r.closed_gap_mask.sum()), 40)
        self.assertEqual(r.parameters['smoothing_radius_px'], 1)
        json.dumps(r.to_dict(), allow_nan=False)

    def test_validation_empty_tiny_and_full(self):
        for key in ('closing_radius_px', 'max_gap_area'):
            for value in (True, -1, 1.0, None, np.int64(2)):
                with self.assertRaises(ValueError):
                    cleanup_mask(notch(), **{key: value})
        with self.assertRaises(ValueError):
            cleanup_mask(notch(), closing_radius_px=6)
        for shape in ((1, 1), (1, 30), (3, 3), (7, 8)):
            for value in (False, True):
                raw = np.full(shape, value, bool)
                r = cleanup_mask(raw, closing_radius_px=5, max_gap_area=10000)
                np.testing.assert_array_equal(r.object_mask, raw)

    def test_numpy_only_execution(self):
        subprocess.run([sys.executable, '-c', '''
import sys
sys.modules['cv2'] = None
import numpy as np
from depth_image_lens import cleanup_mask
m = np.zeros((30, 40), bool); m[5:25, 5:35] = True; m[5:15, 20] = False
r = cleanup_mask(m, closing_radius_px=1, max_gap_area=10)
assert r.closed_gap_mask.sum() == 10
'''], check=True, capture_output=True, text=True)

    def test_random_foreground_components_and_blocking_invariants(self):
        rng = np.random.default_rng(907)
        for _ in range(60):
            raw = rng.random((20, 24)) < rng.uniform(.2, .8)
            observed = rng.random(raw.shape) > .005
            protected = rng.random(raw.shape) < .005
            r = cleanup_mask(raw, closing_radius_px=1, max_gap_area=30,
                             observed_mask=observed, protected_mask=protected)
            self.assertEqual(_component_count(raw), _component_count(r.object_mask))
            self.assertTrue(np.all(r.object_mask[raw]))
            np.testing.assert_array_equal(r.object_mask[~observed | protected], raw[~observed | protected])
            np.testing.assert_array_equal(r.object_mask[:2], raw[:2])
            np.testing.assert_array_equal(r.object_mask[-2:], raw[-2:])


if __name__ == '__main__':
    unittest.main()
