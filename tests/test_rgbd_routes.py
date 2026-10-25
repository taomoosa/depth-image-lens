import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector, SeededColorAssistant
from examples.example_rgbd import apply_color_constraints, run
from .rgbd_route_cases import route_cases
from .rgbd_cases import cases, refinement_negatives


def assist(case,policy):
    return run(case['depth'],case['difference'],case['color'],case['hand_rois'],
        case['object_seeds'],case['hand_seeds'],color_order='RGB',registered_to_depth=True,
        semantic_path_policy=policy)


class SemanticRouteTests(unittest.TestCase):
    def test_default_outputs_preserved_on_existing_regressions(self):
        for c in list(cases())+list(refinement_negatives()):
            baseline=assist(c,'component')[2];route=assist(c,'observed_route')[2]
            self.assertEqual((baseline.bridge,baseline.large_sag,baseline.drape),
                             (route.bridge,route.large_sag,route.drape))

    def test_route_excludes_unknown_without_changing_depth(self):
        for c in route_cases():
            if c['condition'] not in ('unknown_branch','partial_unknown_patch'):continue
            before=c['depth'].copy();base,colors,result=assist(c,'observed_route')
            component=assist(c,'component')[2]
            self.assertEqual(component.bridge,'unknown');self.assertEqual(result.bridge,'yes')
            self.assertTrue(result.features['rgb_semantic_route_used'])
            anchors=base.diagnostics['left_anchor_mask']|base.diagnostics['right_anchor_mask']
            self.assertFalse((result.diagnostics['bridge_corridor_mask']&colors.appearance_uncertain_mask&~anchors).any())
            np.testing.assert_array_equal(result.invalid_mask,base.invalid_mask)
            np.testing.assert_array_equal(c['depth'],before)
            self.assertFalse(result.diagnostics['inferred_gap_mask'].any())
            self.assertEqual(result.drape,component.drape)
            if c['condition']=='partial_unknown_patch':
                self.assertIn('ratio',result.stage_results['sag'])
                self.assertEqual(result.stage_results['sag']['status'],'computed')

    def test_semantic_sensor_and_crop_cuts_remain_unknown(self):
        for c in route_cases():
            if c['condition'] in ('full_semantic_cut','raw_depth_cut','cropped_anchor'):
                self.assertEqual(assist(c,'observed_route')[2].bridge,'unknown')

    def test_robot_shapes_are_not_accepted_as_object_bridges(self):
        for c in route_cases():
            if c['condition'] in ('known_robot_bridge','unknown_robot_bridge'):
                self.assertEqual(assist(c,'observed_route')[2].bridge,'unknown')

    def test_route_policy_requires_no_gap_inference(self):
        c=next(route_cases());d=BridgeDrapeDetector(max_gap_px=1)
        base=d.detect_16uc1(c['depth'],c['difference'],c['hand_rois'],diagnostics=True)
        colors=SeededColorAssistant().segment_16uc1(c['depth'],c['difference'],c['color'],object_seeds=c['object_seeds'],
            hand_seeds=c['hand_seeds'],color_order='RGB',registered_to_depth=True)
        with self.assertRaises(ValueError):apply_color_constraints(d,c['depth'],c['difference'],c['hand_rois'],base,colors,semantic_path_policy='observed_route')


if __name__=='__main__':unittest.main()
