"""Optional region evidence must explain gates without changing selection."""
import unittest
import numpy as np
from depth_image_lens._rgbd_color_region import color_region_fusion


class RegionDiagnosticsTests(unittest.TestCase):
    def run_case(self, *, diagnostics=True, delta=None, valid=None, lab=None, score=None, **settings):
        shape = (10, 12)
        delta = np.full(shape, .02) if delta is None else delta
        valid = np.ones(shape, bool) if valid is None else valid
        lab = np.zeros(shape + (3,)) if lab is None else lab
        score = np.full(shape, 4.) if score is None else score
        p = dict(depth_threshold_m=.01, region_color_relaxation_ratio=.5,
                 region_max_color_groups=8, region_color_tolerance=10.,
                 region_depth_clip_ratio=4., region_min_area=5,
                 region_min_valid_fraction=.5, region_mean_depth_ratio=.5,
                 region_min_depth_fraction=.2)
        p.update(settings)
        fg = valid & (delta > .01)
        return color_region_fusion(delta, fg, valid & (delta < -.01),
            score > 1., np.ones(shape, bool), np.zeros(shape, bool), lab,
            np.ones(shape, bool), {'background_combined_score': score}, valid,
            np.ones(shape, bool), p, diagnostics=diagnostics)

    def test_optional_diagnostics_preserve_outputs_and_existing_diagnostics(self):
        for empty in (False, True):
            score = np.zeros((10, 12)) if empty else None
            off = self.run_case(diagnostics=False, score=score)
            on = self.run_case(score=score)
            for a, b in zip(off[:3], on[:3]):
                np.testing.assert_array_equal(a, b)
            for key, value in off[3].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, on[3][key])
                else:
                    self.assertEqual(value, on[3][key])
            self.assertNotIn('region_group_color_distance', off[3])

    def test_independent_failures_are_not_exclusive(self):
        debug = self.run_case(delta=np.zeros((10, 12)), region_min_area=121)[3]
        self.assertTrue(debug['region_fail_area_mask'].all())
        self.assertTrue(debug['region_fail_mean_depth_mask'].all())
        self.assertTrue(debug['region_fail_depth_fraction_mask'].all())
        self.assertTrue(debug['region_pass_valid_fraction_mask'].all())
        self.assertFalse(debug['region_accepted_mask'].any())

    def test_region_acceptance_does_not_hide_pixel_vetoes(self):
        valid = np.ones((10, 12), bool); valid[0, 0] = False
        delta = np.full((10, 12), .02); delta[0, 1] = -.02
        selected, _, _, d = self.run_case(valid=valid, delta=delta)
        self.assertTrue(d['region_accepted_mask'].all())
        self.assertTrue(d['region_invalid_depth_blocked_mask'][0, 0])
        self.assertTrue(d['region_farther_blocked_mask'][0, 1])
        self.assertFalse(selected[0, :2].any())
        self.assertEqual(int(selected.sum()), 118)

    def test_missing_observation_gates(self):
        d = self.run_case(valid=np.zeros((10, 12), bool))[3]
        for name in ('observed', 'valid_fraction', 'mean_depth', 'depth_fraction'):
            self.assertTrue(d['region_fail_' + name + '_mask'].all())
        self.assertTrue(np.isnan(d['region_clipped_mean_depth_m']).all())

    def test_group_distances_budget_and_empty_outside(self):
        lab = np.zeros((10, 12, 3)); lab[:, 5:8, 0] = 5.; lab[:, 8:, 0] = 30.
        d = self.run_case(lab=lab, region_max_color_groups=1)[3]
        np.testing.assert_array_equal(d['region_group_color_distance'][:, :5], 0.)
        np.testing.assert_array_equal(d['region_group_color_distance'][:, 5:8], 5.)
        self.assertTrue(np.isnan(d['region_group_color_distance'][:, 8:]).all())
        self.assertTrue(d['region_color_budget_exceeded_mask'][:, 8:].all())
        self.assertFalse(d['region_fail_area_mask'][:, 8:].any())
        self.assertEqual(d['region_color_representatives'], [{'id': 1, 'lab': [0., 0., 0.]}])
        self.assertEqual(d['region_thresholds']['region_color_tolerance'], 10.)
