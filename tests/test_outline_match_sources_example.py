"""Executable source adapters preserve observation and component ambiguity."""
import json
import unittest
import numpy as np
from depth_image_lens import PartialOutlineMatcher, OutlineArticulation
from examples.example_outline_match_sources import (
    make_scene, run_examples, match_mask, match_exteriors, observed_exterior_edges,
)


class OutlineMatchSourcesExampleTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_scene()
        self.options = dict(matcher=PartialOutlineMatcher(),
                            articulation=OutlineArticulation(2,3,(62.5,50)))

    def test_all_documented_source_routes_execute(self):
        out = run_examples()
        expected = {
            'bool_mask', 'closed_xy', 'OuterContourResult',
            'OuterContourResult.component_mask', 'BoundaryResult.object_mask',
            'RGBDBackgroundResult', 'CornerResult.object_mask', 'ColorAssistResult',
            'ReferenceSegmentationResult', 'BoundaryProximityResult', 'SurfaceResult',
            'EdgeBulgeResult', 'PickCandidateResult', 'BridgeDrapeResult',
            'analyze_color_geometry -> BridgeDrapeResult',
            'LiftResult.diagnostics.object_mask',
            'GraspResult.diagnostics.eligible_object_mask',
            'ObjectOnlyGraspDetector retained INPUT mask',
        }
        self.assertEqual(set(out['reports']),expected)
        for name, report in out['reports'].items():
            with self.subTest(source=name):
                self.assertEqual(report.status,'matched')
                self.assertTrue(report.best['supported'])
                json.dumps(report.to_dict(),allow_nan=False)
        self.assertEqual(set(out['per_contour']),{
            'BoundaryResult.exterior','CornerResult.diagnostics.contours_xy',
            'OuterContourResult.contours.points_xy'})
        for name, rows in out['per_contour'].items():
            with self.subTest(source=name):
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['result'].status,'matched')

    def test_long_simplified_edge_cannot_skip_missing_middle(self):
        s = self.scene
        observed = s['observed'].copy()
        observed[100,60] = False  # Middle of the 85px hem-to-armpit edge.
        edges = observed_exterior_edges(s['polygon'],observed)
        self.assertFalse(edges[0])
        self.assertTrue(edges[1:].all())
        self.assertTrue(observed[155,60] and observed[70,60])
        row = match_exteriors(s['target'],[s['polygon']],observed_mask=observed,**self.options)[0]
        self.assertEqual(row['result'].status,'unknown')

    def test_frame_contacts_are_not_fully_observed(self):
        s = self.scene
        points = s['polygon'] - [20,0]
        edges = observed_exterior_edges(points,s['observed'])
        self.assertFalse(edges[2])
        self.assertFalse(edges[3])
        row = match_exteriors(s['target'],[points],observed_mask=s['observed'],**self.options)[0]
        self.assertEqual(row['result'].status,'unknown')

    def test_joint_vs_per_contour_ambiguity(self):
        s = self.scene
        double = np.concatenate([s['object_mask'],s['object_mask']],axis=1)
        observed = np.ones_like(double)
        joint = match_mask(s['target'],double,observed_mask=observed,**self.options)
        self.assertEqual(joint.status,'ambiguous')
        separate = match_exteriors(s['target'],[s['polygon'],s['polygon']+[260,0]],
                                   observed_mask=observed,**self.options)
        self.assertEqual([row['result'].status for row in separate],['matched','matched'])
        self.assertEqual([row['source_index'] for row in separate],[0,1])

    def test_source_status_and_tiny_contours_are_not_promoted(self):
        s = self.scene
        row = match_exteriors(s['target'],[s['polygon']],observed_mask=s['observed'],
                              source_statuses=['ambiguous_self_touch'],**self.options)[0]
        self.assertIsNone(row['result'])
        self.assertIn('ambiguous_self_touch',row['reason'])
        row = match_exteriors(s['target'],[[[1,1],[2,2]]],
                              observed_mask=s['observed'],**self.options)[0]
        self.assertIsNone(row['result'])
        self.assertEqual(row['reason'],'fewer_than_three_vertices')

    def test_edge_helper_validation_and_inputs_unchanged(self):
        s = self.scene
        points, observed = s['polygon'].copy(), s['observed'].copy()
        self.assertTrue(observed_exterior_edges(points,observed).all())
        np.testing.assert_array_equal(points,s['polygon'])
        np.testing.assert_array_equal(observed,s['observed'])
        with self.assertRaises(ValueError):
            observed_exterior_edges(points+.5,observed)
        with self.assertRaises(ValueError):
            observed_exterior_edges(points,observed.astype(np.uint8))
        outside = points.copy()
        outside[3,0] = -1
        edges = observed_exterior_edges(outside,observed)
        self.assertFalse(edges[2])
        self.assertFalse(edges[3])


if __name__ == '__main__':
    unittest.main()
