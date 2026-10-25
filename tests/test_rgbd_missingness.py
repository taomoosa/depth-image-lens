"""Checks for opt-in local support, diagnostics, and conservative failure cases."""
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from depth_image_lens.rgbd_diagnostics import uncertainty_report
from examples.example_rgbd import run, apply_color_constraints
from tests.rgbd_missingness_cases import validation_cases


class MissingnessTests(unittest.TestCase):
    def case(self, name):
        return next(c for c in validation_cases() if c['condition'] == name)

    def detect(self, case, **kw):
        return BridgeDrapeDetector().detect_16uc1(case['depth'], case['difference'], case['hand_rois'], diagnostics=True, **kw)

    def test_local_observed_patch_recovers_positive_without_filling(self):
        c = self.case('anchor_object_1'); original = c['depth'].copy()
        old = self.detect(c); new = self.detect(c, anchor_policy='local_observed')
        self.assertEqual((old.bridge, new.bridge), ('yes', 'yes'))
        self.assertEqual(old.features['conservative_decisions']['bridge'],'unknown')
        self.assertFalse(new.features['anchor_observable'][0])
        self.assertEqual(new.features['inferred_depth_pixels'], 0)
        self.assertFalse((new.work_mask & new.invalid_mask).any())
        self.assertFalse(new.diagnostics['inferred_gap_mask'].any())
        np.testing.assert_array_equal(c['depth'], original)
        self.assertEqual(new.drape, old.drape)

    def test_full_cuts_missing_anchor_and_exclusion_remain_unknown(self):
        for name in ('object_cut_1', 'object_cut_5', 'anchor_missing', 'excluded_anchor', 'anchor_object_2'):
            c = self.case(name)
            self.assertEqual(self.detect(c, anchor_policy='local_observed', **c['options']).bridge, 'unknown', name)

    def test_hidden_partial_crop_and_roi_never_restored(self):
        c = self.case('anchor_background_1')
        for visibility in (('hidden', 'visible'), ('partial', 'visible')):
            result=self.detect(c, anchor_policy='local_observed', hand_visibility=visibility)
            self.assertEqual(result.bridge,'yes')
            self.assertEqual(result.stage_results['grasp']['value'],'unknown')
            self.assertTrue(any('hand_' in w for w in result.stage_results['observed_bridge']['warnings']))
        self.assertEqual(self.detect(self.case('crop'), anchor_policy='local_observed').bridge, 'unknown')
        roi = np.ones(c['depth'].shape, bool); roi[:, :35] = False
        self.assertEqual(self.detect(c, anchor_policy='local_observed', work_roi=roi).bridge, 'unknown')

    def test_zero_difference_not_sensor_hole_or_recovered_contact(self):
        c = self.case('clean'); c['difference'][24:32, 20:36] = 0
        result = self.detect(c, anchor_policy='local_observed')
        self.assertEqual(result.bridge, 'no')  # Complete observed exclusion, not inferred depth.
        self.assertFalse(result.invalid_mask.any())
        self.assertFalse(result.features['local_anchor_positive_used'])

    def test_no_optional_gap_inference_with_local_policy(self):
        c = self.case('clean')
        with self.assertRaises(ValueError):
            BridgeDrapeDetector(max_gap_px=1).detect_16uc1(c['depth'], c['difference'], c['hand_rois'], anchor_policy='local_observed')
        with self.assertRaises(ValueError): self.detect(c, anchor_policy='invalid')

    def test_semantics_still_gate_robot_and_color_invalid_paths(self):
        for name in ('robot_false_bridge', 'robot_unseen_color', 'similar_color', 'color_invalid_path'):
            c = self.case(name)
            base, colors, _ = run(c['depth'], c['difference'], c['color'], c['hand_rois'], c['object_seeds'], c['hand_seeds'],
                                  color_order='RGB', registered_to_depth=True, **c['options'])
            opts = {k: v for k, v in c['options'].items() if k != 'color_valid_mask'}
            r = apply_color_constraints(BridgeDrapeDetector(), c['depth'], c['difference'], c['hand_rois'], base, colors,
                                        local_anchor_support=True, **opts)
            self.assertEqual(r.bridge, 'unknown', name)

    def test_cause_masks_separate_remote_holes_guard_and_exclusions(self):
        c = self.case('remote_holes')
        base, colors, final = run(c['depth'], c['difference'], c['color'], c['hand_rois'], c['object_seeds'], c['hand_seeds'],
                                  color_order='RGB', registered_to_depth=True)
        report, masks = uncertainty_report(c['depth'], c['difference'], final, colors)
        cause = report['causes']['sensor_zero_depth']
        self.assertEqual(cause['pixels'], 120)
        self.assertEqual(cause['regions']['search'], 0)
        self.assertEqual(cause['regions']['outside_search'], 120)
        self.assertEqual(final.bridge, 'yes')
        self.assertFalse((masks['color_boundary_guard_only'] & masks['sensor_zero_depth']).any())
        self.assertFalse((masks['zero_difference_background_or_excluded'] & masks['sensor_zero_depth']).any())

    def test_background_and_observation_masks_not_raw_sensor_holes(self):
        for name in ('unobserved_anchor', 'background_invalid_anchor'):
            c = self.case(name)
            base, colors, final = run(c['depth'], c['difference'], c['color'], c['hand_rois'], c['object_seeds'], c['hand_seeds'],
                                      color_order='RGB', registered_to_depth=True, **c['options'])
            report, masks = uncertainty_report(c['depth'], c['difference'], final, colors, **c['options'])
            self.assertFalse(masks['sensor_zero_depth'].any())
            cause = 'caller_unobserved' if name.startswith('unobserved') else 'background_unavailable'
            self.assertEqual(report['causes'][cause]['regions']['left_anchor'], 9)


if __name__ == '__main__': unittest.main()
