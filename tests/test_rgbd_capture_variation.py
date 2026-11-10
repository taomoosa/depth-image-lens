"""Repeat-capture counterexamples, not empirical sensor accuracy measurements."""
import unittest

import numpy as np

from depth_image_lens import RGBDBackgroundSegmenter, background_difference
from .test_rgbd_background import scene, detect


class RGBDCaptureVariationTests(unittest.TestCase):
    def bounded(self, color, depth, empty_color, empty_depth, upper_mm=80.):
        delta, valid = background_difference(
            depth, empty_depth, depth_unit='mm', background_unit='mm')
        roi = ~(valid & (delta > upper_mm))
        return detect(color, depth, empty_color, empty_depth,
                      RGBDBackgroundSegmenter(fusion_mode='depth_required',
                                              missing_depth_policy='unknown'),
                      work_roi=roi)

    def test_common_z_translation_cancels_only_with_matching_reference(self):
        # Ideal frontoparallel arrays, without rescaling/parallax/noise changes.
        for table_z in (700., 1000., 1600.):
            c, _, bc, _ = scene()
            bz = np.full(c.shape[:2], table_z)
            z = bz.copy(); z[10:30, 10:40] -= 20.
            expected = z < bz
            np.testing.assert_array_equal(self.bounded(c, z, bc, bz).object_mask,
                                          expected)

    def test_color_reference_check_does_not_validate_depth_drift(self):
        c, z, bc, bz = scene()
        z = z.astype(float) - 7.  # Old empty reference or coherent sensor bias.
        check = np.ones(z.shape, bool)
        r = detect(c, z, bc, bz,
                   RGBDBackgroundSegmenter(fusion_mode='depth_required'),
                   background_check_mask=check)
        self.assertNotIn(r.stages['color_reference']['status'],
                         ('rejected', 'insufficient'))
        self.assertTrue(r.object_mask.all())

    def test_same_physical_height_changes_camera_z_band_with_tilt(self):
        # Table n.X=d; raised parallel surface n.X=d-h, in millimetres.
        # On each optical ray r=(x/f,y/f,1), dZ=h/(n.r), NOT h*cos(theta).
        c, _, bc, _ = scene()
        x = (np.indices(c.shape[:2])[1] - 30.) / 600.
        target = np.zeros(x.shape, bool); target[10:30, 20:40] = True
        for angle, accepted in ((0., True), (45., False)):
            theta = np.deg2rad(angle)
            denominator = np.sin(theta) * x + np.cos(theta)
            bz = 1000. * np.cos(theta) / denominator
            z = bz.copy(); z[target] -= 60. / denominator[target]
            r = self.bounded(c, z, bc, bz)
            self.assertEqual(bool(r.object_mask[target].all()), accepted)
            if not accepted:
                self.assertFalse(r.object_mask.any())
                self.assertFalse(r.outline_observed_mask[target].any())

    def test_individually_separable_captures_can_have_no_shared_upper_bound(self):
        # A needs upper>=100 and <130; B needs upper>=20 and <60.
        # Their admissible upper-bound intervals do not intersect.
        for upper in (40., 80., 120.):
            solved = []
            for target_delta, unwanted_delta in ((100., 130.), (20., 60.)):
                c, z, bc, bz = scene(); z = z.astype(float)
                target = np.zeros(z.shape, bool); target[10:25, 10:30] = True
                unwanted = np.zeros(z.shape, bool); unwanted[5:10, 40:50] = True
                z[target] -= target_delta; z[unwanted] -= unwanted_delta
                r = self.bounded(c, z, bc, bz, upper)
                solved.append(np.array_equal(r.object_mask, target))
            self.assertFalse(all(solved))

    def test_overlapping_depth_errors_cannot_be_fixed_by_lower_threshold(self):
        # Thin target reports 2 mm; empty shadow reports a coherent 4 mm error.
        # Any lower threshold selecting the former also admits the latter.
        c, z, bc, bz = scene(); z = z.astype(float)
        target = np.zeros(z.shape, bool); target[10:25, 10:30] = True
        shadow = np.zeros(z.shape, bool); shadow[10:25, 30:45] = True
        z[target] -= 2.; z[shadow] -= 4.
        for threshold in (.0005, .0015, .0025, .0035, .005):
            r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(
                fusion_mode='depth_required', depth_threshold_m=threshold,
                depth_margin_m=0.))
            self.assertFalse(r.object_mask[target].all()
                             and not r.object_mask[shadow].any())

    def test_one_rgb_count_can_create_or_remove_every_seed(self):
        # Encoded 8-bit Lab quantization straddles the default strict dC>24 gate.
        # The fixture is a counterexample, not a measurement of sensor jitter.
        for blue, count in ((182, 0), (183, 600)):
            for steps in (0, 2, 20):
                c, z, bc, bz = scene()
                c[10:30, 10:40] = (110, 145, blue)
                z[10:30, 10:40] = 980
                r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(
                    fusion_mode='color_seeded', color_growth_max_steps=steps))
                self.assertEqual(int(r.object_mask.sum()), count)

    def test_fixed_growth_pixels_do_not_recover_arbitrarily_wide_weak_band(self):
        # Projected weak-edge width changes with imaging scale; keep settings fixed.
        for width, full_recovery in ((1, True), (2, True), (4, False)):
            c, z, bc, bz = scene()
            c[10:30, 10:20] = (30, 150, 210)  # Strict chroma seed.
            c[10:30, 20:20+width] = (110, 145, 165)  # Relaxed-only evidence.
            z[10:30, 10:20+width] = 980
            r = detect(c, z, bc, bz,
                       RGBDBackgroundSegmenter(fusion_mode='color_seeded'))
            self.assertEqual(bool(r.object_mask[10:30, 20:20+width].all()),
                             full_recovery)
            self.assertEqual(int(r.object_mask.sum()), 200 + 20 * min(width, 2))

    def test_uncertainty_margin_does_not_change_depth_selection(self):
        c, z, bc, bz = scene(); z = z.astype(float)
        z[10:30, 10:40] = 994.5
        masks = []
        uncertainty = []
        for margin in (0., .001, .004):
            r = detect(c, z, bc, bz, RGBDBackgroundSegmenter(
                fusion_mode='depth_required', depth_margin_m=margin))
            masks.append(r.object_mask)
            uncertainty.append(int(r.uncertain_mask.sum()))
        for mask in masks[1:]:
            np.testing.assert_array_equal(mask, masks[0])
        self.assertEqual(uncertainty, [0, 600, 600])


if __name__ == '__main__':
    unittest.main()
