"""Default-policy regression: color uncertainty is not sensor missingness."""
import unittest
import numpy as np
from examples.example_rgbd import run
from .rgbd_cases import cases, refinement_negatives


def evaluate(case, **kwargs):
    return run(case['depth'], case['difference'], case['color'], case['hand_rois'],
               case['object_seeds'], case['hand_seeds'], color_order='RGB',
               registered_to_depth=True, **kwargs)


class ColorIntegrationTests(unittest.TestCase):
    def test_default_guard_preserves_observed_anchor_depth(self):
        case = next(cases())
        base, colors, revised = evaluate(case)
        self.assertEqual(colors.summary['boundary_guard_px'], 1)
        self.assertTrue(colors.uncertain_mask.any())
        self.assertFalse(colors.appearance_uncertain_mask.any())
        self.assertEqual((revised.bridge, revised.large_sag, revised.drape), ('yes','yes','both'))
        np.testing.assert_array_equal(base.invalid_mask, revised.invalid_mask)
        np.testing.assert_array_equal(base.foreground_mask, revised.foreground_mask)
        self.assertFalse(revised.features['rgb_physical_contact_verified'])
        self.assertEqual(revised.features['rgb_semantic_uncertain_pixels'],0)
        self.assertGreater(revised.features['rgb_guard_only_uncertain_pixels'],0)

    def test_true_missingness_and_zero_gap_are_not_repaired(self):
        for kind in ('raw_zero','observed_false','background_invalid','known_zero_gap'):
            with self.subTest(kind=kind):
                case = next(cases()); opts = {}
                if kind == 'raw_zero': case['depth'][:,59:62] = 0
                elif kind == 'known_zero_gap': case['difference'][:,59:62] = 0
                else:
                    valid = np.ones_like(case['object_gt']); valid[:,59:62] = False
                    opts['observed_mask' if kind=='observed_false' else 'background_valid'] = valid
                base, _, revised = evaluate(case, **opts)
                self.assertNotEqual(revised.bridge,'yes')
                np.testing.assert_array_equal(base.invalid_mask,revised.invalid_mask)
                np.testing.assert_array_equal(base.foreground_mask,revised.foreground_mask)
                if kind == 'known_zero_gap':
                    self.assertFalse(revised.foreground_mask[:,59:62].any())
                else: self.assertTrue(revised.invalid_mask[:,59:62].all())

    def test_negative_appearance_gate_does_not_hide_depth_geometry(self):
        for case in refinement_negatives():
            with self.subTest(name=case['name']):
                base, _, revised = evaluate(case)
                self.assertEqual(base.bridge,'yes')  # Known semantic false positive.
                self.assertEqual((revised.bridge,revised.drape),('unknown','unknown'))
                self.assertFalse(revised.diagnostics['sag_path_used'])
                np.testing.assert_array_equal(base.invalid_mask,revised.invalid_mask)
                if case['name']=='unseeded_arm_color_negative':
                    self.assertEqual(revised.features['rgb_depth_geometry_before_semantic_gate']['bridge'],'yes')
                    self.assertEqual(revised.reasons['bridge'],'rgb_unresolved_appearance_on_depth_path')

    def test_caller_arm_exclusion_is_not_cleared_at_anchors(self):
        case = next(cases()); arm = np.zeros_like(case['object_gt']); arm[:24] = True
        base, _, revised = evaluate(case, arm_mask=arm)
        self.assertEqual(base.bridge,'unknown'); self.assertEqual(revised.bridge,'unknown')
        self.assertEqual(revised.features['anchor_observable'],[False,False])



if __name__ == '__main__': unittest.main()
