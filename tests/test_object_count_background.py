"""Synthetic external-mask background veto regression tests."""
import unittest
from unittest.mock import patch
import numpy as np
from depth_image_lens import OverheadObjectCounter, RGBDBackgroundSegmenter


class MaskBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((60, 80, 3), (150, 140, 100), np.uint8)
        self.color = self.background.copy()
        self.truth = np.zeros((60, 80), bool)
        self.truth[15:40, 20:50] = True
        self.mask = np.zeros_like(self.truth)
        self.mask[11:44, 16:54] = True
        self.color[self.truth] = (210, 35, 35)
        self.empty = np.full(self.truth.shape, 1000, np.uint16)
        self.depth = self.empty.copy()
        self.depth[self.truth] = 980

    def run_scene(self, counter=None, **kwargs):
        args = dict(color_order='RGB', background_color=self.background,
                    depth=self.depth, background_depth=self.empty,
                    depth_scale_m=.001, registered_to_depth=True, diagnostics=True)
        args.update(kwargs)
        return (counter or OverheadObjectCounter()).detect_mask(self.color, self.mask, **args)

    def test_padding_removed_without_changing_true_pixels_or_inputs(self):
        original = self.mask.copy()
        r = self.run_scene()
        self.assertEqual(r.count, 1)
        np.testing.assert_array_equal(r.object_mask, self.truth)
        np.testing.assert_array_equal(r.diagnostics['background_removed_mask'], self.mask & ~self.truth)
        np.testing.assert_array_equal(original, self.mask)
        self.assertEqual(r.metrics['background_removed_pixels'], 504)

    def test_raised_matching_color_retained(self):
        self.color[self.truth] = self.background[self.truth]
        r = self.run_scene()
        self.assertEqual(r.count, 1)
        np.testing.assert_array_equal(r.object_mask, self.truth)

    def test_thin_matching_color_all_removed_abstains(self):
        self.color[:] = self.background
        self.depth[self.truth] = 999
        r = self.run_scene()
        self.assertIsNone(r.count)
        self.assertIn('background_matching_mask_may_include_thin_objects', r.reasons)

    def test_missing_depth_keeps_candidates(self):
        self.depth[self.mask] = 0
        r = self.run_scene()
        np.testing.assert_array_equal(r.candidate_mask, self.mask)
        self.assertIsNone(r.count)
        self.assertFalse(r.diagnostics['background_removed_mask'].any())

    def test_color_only_no_fake_depth(self):
        r = self.run_scene(depth=None, background_depth=None, depth_scale_m=None)
        np.testing.assert_array_equal(r.candidate_mask, self.mask)
        self.assertIsNone(r.count)

    def test_no_reference_unchanged_and_no_segmenter_call(self):
        c = OverheadObjectCounter()
        with patch.object(c.background_segmenter, 'detect', side_effect=AssertionError):
            r = self.run_scene(c, background_color=None)
            s = self.run_scene(c)
        self.assertEqual(r.candidate_count, 2)
        self.assertEqual(s.candidate_count, 1)
        self.assertNotIn('background_removed_mask', r.diagnostics)

    def test_bgr_and_crop(self):
        r = self.run_scene()
        self.color = self.color[..., ::-1]
        self.background = self.background[..., ::-1]
        s = self.run_scene(color_order='BGR')
        np.testing.assert_array_equal(r.label_map, s.label_map)
        roi = np.zeros_like(self.mask); roi[:30] = True
        t = self.run_scene(color_order='BGR', work_roi=roi)
        self.assertFalse(t.object_mask[30:].any())
        self.assertTrue(t.groups[0]['touches_crop'])

    def test_inferred_and_invalid_reference_not_removed(self):
        invalid = ~self.truth
        r = self.run_scene(background_color_valid_mask=~invalid)
        self.assertFalse(r.diagnostics['background_removed_mask'].any())
        for name in ('uncertain_mask', 'cleanup_changed_mask'):
            r = self.run_scene(**{name: self.mask})
            self.assertFalse(r.diagnostics['background_removed_mask'].any())
            self.assertIsNone(r.count)

    def test_rejected_reference_does_not_suppress(self):
        self.background[:] = 20
        check = ~self.mask
        r = self.run_scene(background_check_mask=check)
        self.assertEqual(r.metrics['color_reference_status'], 'rejected')
        self.assertFalse(r.diagnostics['background_removed_mask'].any())
        self.assertIsNone(r.count)

    def test_validation(self):
        for kw in (dict(background_color=self.background[:-1]),
                   dict(background_color=self.background.astype(float)),
                   dict(background_color=None, background_check_mask=self.mask),
                   dict(background_color_valid_mask=np.ones((2, 2), bool))):
            with self.assertRaises(ValueError): self.run_scene(**kw)
        for value in (0, -1, True, np.nan, np.inf):
            with self.assertRaises(ValueError):
                OverheadObjectCounter(background_match_luma_tolerance=value)


    def test_compensation_on_declared_table(self):
        self.color = np.clip(self.color.astype(int) + 8, 0, 255).astype(np.uint8)
        c = OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(
            background_compensate=True))
        r = self.run_scene(c, background_check_mask=~self.mask)
        self.assertEqual(r.count, 1)
        self.assertTrue(r.parameters['background_mask_filter']['compensation_applied'])
        np.testing.assert_array_equal(r.object_mask, self.truth)

    def test_dark_shadow_is_not_assumed_background(self):
        rim = self.mask & ~self.truth
        self.color[rim] //= 2
        r = self.run_scene()
        self.assertFalse(r.diagnostics['background_removed_mask'].any())
        self.assertIsNone(r.count)

    def test_excluded_region_not_reported_as_removed(self):
        r = self.run_scene(exclusion_mask=self.mask)
        self.assertFalse(r.diagnostics['background_removed_mask'].any())
        self.assertEqual(r.metrics['background_removed_pixels'], 0)
