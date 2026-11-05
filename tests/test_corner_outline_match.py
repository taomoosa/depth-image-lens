"""Synthetic contracts for corner registration, not real garment accuracy."""
import json
import unittest
import numpy as np

from depth_image_lens import (
    CornerOutlineMatcher, PartialOutlineMatcher, OutlineArticulation, OuterContourDetector,
)
from .test_outline_match import shirt, sleeve, mask_of


def rotation(degrees):
    theta = np.radians(degrees)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


class CornerOutlineTests(unittest.TestCase):
    def setUp(self):
        self.target, self.current = shirt()
        self.matcher = CornerOutlineMatcher()
        self.art = OutlineArticulation(2, 3, (62.5, 50))

    def test_identity_parameters_and_per_edge_outputs(self):
        result = self.matcher.match_contour(self.target, self.current)
        self.assertEqual(result.status, 'matched')
        self.assertLess(result.best['score'], 1e-10)
        self.assertEqual(result.diagnostics['target_corner_indices'], [1, 2, 3, 4])
        self.assertEqual(len(result.best['edges']), 5)
        for key in ('scale', 'scale_x', 'scale_y', 'axis_ratio'):
            self.assertAlmostEqual(result.best[key], 1.)
        for edge in result.best['edges']:
            self.assertTrue(edge['passed'])
            self.assertAlmostEqual(edge['length_ratio'], 1.)
        self.assertEqual(result.best['edges'][0]['role'], 'endpoint_tail')
        self.assertEqual(result.best['edges'][2]['role'], 'between_corners')

    def test_pose_axial_scale_and_cyclic_order(self):
        for sx, sy, degrees, reverse, roll in [(1.15, .95, 63, False, 4),
                                              (.7, .7, -133, True, 9), (1.7, 1.6, 15, True, 0)]:
            with self.subTest(sx=sx, sy=sy, degrees=degrees):
                q = (self.current*[sx, sy])@rotation(degrees).T+[300, 150]
                q = np.roll(q[::-1] if reverse else q, roll, axis=0)
                r = self.matcher.match_contour(self.target, q)
                self.assertEqual(r.status, 'matched')
                self.assertAlmostEqual(r.best['scale_x'], sx, places=7)
                self.assertAlmostEqual(r.best['scale_y'], sy, places=7)
                self.assertAlmostEqual(r.best['rotation_deg'], degrees, places=7)
                self.assertLess(r.best['geometry_score'], 1e-8)

    def test_declared_target_axes_and_origin(self):
        alpha = 37
        target = self.target@rotation(alpha).T+[70, -40]
        basis = rotation(alpha)
        matrix = rotation(23)@basis@np.diag([1.15, .95])@basis.T
        current = (self.current@basis.T+[70, -40])@matrix.T+[120, 170]
        r = CornerOutlineMatcher(axis_angle_deg=alpha).match_contour(target, current)
        self.assertEqual(r.status, 'matched')
        self.assertAlmostEqual(r.best['rotation_deg'], 23, places=7)
        np.testing.assert_allclose([r.best['scale_x'], r.best['scale_y']], [1.15, .95], atol=1e-8)
        shifted = CornerOutlineMatcher(axis_angle_deg=alpha).match_contour(target-target[0], current)
        self.assertAlmostEqual(r.best['score'], shifted.best['score'])

    def test_endpoints_do_not_determine_transform(self):
        t = self.target.copy()
        t[0] = [60, 130]  # non-corner on the torso edge
        t[-1] = [80, 33]  # non-corner between shoulder and neck
        r = self.matcher.match_contour(t, self.current)
        self.assertEqual(r.status, 'matched')
        np.testing.assert_allclose([r.best['start_xy'], r.best['end_xy']], t[[0, -1]], atol=1e-8)
        np.testing.assert_allclose(r.best['matrix_xy'], np.eye(2), atol=1e-8)
        # Deliberately wrong tail: unchanged corner fit, rejected edge.
        t[0] += [35, 0]
        wrong = self.matcher.match_contour(t, self.current)
        self.assertEqual(wrong.status, 'not_matched')
        self.assertLess(wrong.best['metrics']['corner_rms'], 1e-8)
        self.assertGreater(wrong.best['edges'][0]['score'], 1)

    def test_three_corner_dense_same_mask_and_reverse(self):
        outline = OuterContourDetector(corner_turn_deg=35).detect(mask_of(self.current))
        c = outline.largest_contour()
        points = np.asarray(c['points_xy'], float)
        corners = {k['id']: k for k in c['corners']}
        a, b = corners[6]['point_index'], corners[0]['point_index']
        target = points[(a+np.arange((b-a) % len(points)+1)) % len(points)]
        for t in (target, target[::-1]):
            r = outline.match_partial(t, matcher=self.matcher)
            self.assertEqual(r.status, 'matched')
            self.assertEqual(r.diagnostics['target_corner_count'], 3)
            # Open/closed RDP cuts and tied pixel corners differ slightly by direction.
            self.assertLess(r.best['geometry_score'], .5)

    def test_raster_and_continuous_sleeve_rotation(self):
        for angle in (-20, 0, 20, 30):
            for raster in (False, True):
                with self.subTest(angle=angle, raster=raster):
                    current = sleeve(self.current, angle)
                    if raster:
                        r = OuterContourDetector().detect(mask_of(current)).match_partial(
                            self.target, matcher=self.matcher, articulation=self.art)
                    else:
                        r = self.matcher.match_contour(self.target, current, articulation=self.art)
                    self.assertEqual(r.status, 'matched')
                    self.assertEqual(r.best['articulation_angle_deg'], angle)

    def test_joint_articulation_axis_scale_and_global_rotation(self):
        current = (sleeve(self.current, 20)*[1.1, .95])@rotation(40).T+[150, 200]
        r = self.matcher.match_contour(self.target, current, articulation=self.art)
        self.assertEqual(r.status, 'matched')
        np.testing.assert_allclose([r.best['scale_x'], r.best['scale_y']], [1.1, .95], atol=1e-8)
        self.assertAlmostEqual(r.best['rotation_deg'], 40)
        self.assertEqual(r.best['articulation_angle_deg'], 20)

    def test_transform_amounts_are_independent_guards(self):
        for q, reason in [(self.current*[1.6, .7], 'axis_ratio_exceeds_limit'),
                          (self.current*2.5, 'scale_x_exceeds_limit')]:
            r = self.matcher.match_contour(self.target, q)
            self.assertEqual(r.status, 'not_matched')
            self.assertLess(r.best['geometry_score'], 1e-8)
            self.assertIn(reason, r.best['reasons'])
        r = CornerOutlineMatcher(max_rotation_deg=10).match_contour(self.target, self.current@rotation(40).T)
        self.assertEqual(r.status, 'not_matched')
        self.assertIn('rotation_exceeds_limit', r.best['reasons'])

    def test_fold_missing_extra_corner_shear_and_reflection(self):
        cases = [sleeve(self.current, 90), np.delete(self.current, 2, axis=0),
                 np.vstack([self.current[:2], [[48, 58], [40, 75]], self.current[2:]]),
                 self.current@np.array([[1., .5], [0, 1.]]).T,
                 np.vstack([self.target, [[170, 10], [180, 180]]])*[-1, 1]]
        for i, q in enumerate(cases):
            with self.subTest(case=i):
                r = self.matcher.match_contour(self.target, q, articulation=self.art)
                self.assertEqual(r.status, 'not_matched')

    def test_curve_between_matching_corners_is_not_ignored(self):
        # A smooth bulge on a tail, retaining all anatomical corner positions.
        u = np.linspace(0, 1, 85)
        edge = self.current[0]*(1-u[:, None])+self.current[1]*u[:, None]
        edge[:, 0] += 16*np.sin(np.pi*u)**2
        current = np.vstack([edge[:-1], self.current[1:]])
        r = self.matcher.match_contour(self.target, current)
        self.assertEqual(r.status, 'not_matched')
        self.assertGreater(r.best['limit_ratios']['edges'], 1.)

    def test_zigzag_rejected(self):
        u = np.linspace(0, 1, 61)
        edge = self.current[0]*(1-u[:, None])+self.current[1]*u[:, None]
        edge[:, 0] += 6*np.sin(u*14*np.pi)*np.sin(u*np.pi)**2
        current = np.vstack([edge[:-1], self.current[1:]])
        r = self.matcher.match_contour(self.target, current, articulation=self.art)
        self.assertEqual(r.status, 'not_matched')

    def test_too_few_or_nearly_collinear_corners_withhold_fit(self):
        for target in (self.target[:4], [[0,0], [30,0], [31,1], [60,0], [90,0]]):
            r = self.matcher.match_contour(target, self.current)
            self.assertEqual(r.status, 'unknown')
            self.assertIsNone(r.best)
        # All three strong interior turns occur on an almost horizontal line.
        target = [[0, 40], [20, 0], [60, 1], [40, 0], [50, -30]]
        r = CornerOutlineMatcher(corner_spacing_px=.1, corner_tolerance_px=.1).match_contour(target, self.current)
        self.assertEqual(r.status, 'unknown')
        self.assertIn('interior_corners_are_nearly_collinear', r.diagnostics['reasons'])

    def test_missing_edges_and_clipped_mask(self):
        # A subpixel missing edge between sampled points must not be ignored.
        current = np.vstack([self.current[:3], [[25.01, 72.525], [25, 72.5]], self.current[3:]])
        observed = np.ones(len(current), bool); observed[3] = False
        r = self.matcher.match_contour(self.target, current, observed_edges=observed)
        self.assertEqual(r.status, 'unknown')
        self.assertFalse(r.best['supported'])
        observed = np.ones(len(self.current), bool); observed[8] = False
        r = self.matcher.match_contour(self.target, self.current, observed_edges=observed)
        self.assertEqual(r.status, 'matched')
        mask = mask_of(self.current); observed = np.ones_like(mask); observed[:, :32] = False
        r = OuterContourDetector().detect(mask, observed_mask=observed).match_partial(self.target, matcher=self.matcher)
        self.assertEqual(r.status, 'unknown')

    def test_multiple_objects_are_ambiguous(self):
        mask = mask_of(self.current, (230,480)) | mask_of(self.current+[240,0], (230,480))
        r = OuterContourDetector().detect(mask).match_partial(self.target, matcher=self.matcher)
        self.assertEqual(r.status, 'ambiguous')
        self.assertIn('multiple_distinct_visible_arcs_fit', r.diagnostics['reasons'])

    def test_global_and_local_rotation_confounding_is_reported(self):
        target = np.array([[0,-1], [40,0], [40,40], [0,40], [0,1]], float)
        current = np.vstack([target, [[-5,0]]])
        art = OutlineArticulation(1, 3, (0., 0.))
        r = CornerOutlineMatcher(fraction_range=(.1,.99)).match_contour(target, current, articulation=art)
        self.assertEqual(r.status, 'ambiguous')
        self.assertIn('multiple_transform_explanations_fit', r.diagnostics['reasons'])
        self.assertTrue(r.diagnostics['alternative_transforms'])

    def test_budget_empty_and_topology_withhold_decision(self):
        r = CornerOutlineMatcher(max_hypotheses=16).match_contour(self.target, self.current, articulation=self.art)
        self.assertEqual(r.status, 'unknown')
        self.assertIn('search_budget_exceeded', r.diagnostics['reasons'])
        r = OuterContourDetector().detect(np.zeros((100,100), bool)).match_partial(self.target, matcher=self.matcher)
        self.assertEqual(r.status, 'unknown')
        r = self.matcher.match_contour(self.target, [[0,0], [100,100], [0,100], [100,0]])
        self.assertEqual(r.status, 'unknown')
        r = CornerOutlineMatcher(min_arc_length_px=1000).match_contour(self.target, self.current)
        self.assertEqual(r.status, 'unknown')

    def test_source_corner_indices_match_reported_points_when_reversed(self):
        for current in (self.current, np.roll(self.current[::-1], 4, axis=0)):
            result = self.matcher.match_contour(self.target, current)
            self.assertEqual(result.status, 'matched')
            np.testing.assert_allclose(current[result.best['observed_corner_indices']],
                                       result.best['observed_corner_xy'], atol=1e-8)

    def test_legacy_api_unchanged_and_debug_json_immutability(self):
        before = self.current.copy()
        outline = OuterContourDetector().detect(mask_of(self.current))
        self.assertEqual(outline.match_partial(self.target).status, 'matched')
        r = outline.match_partial(self.target, matcher=self.matcher)
        json.dumps(r.to_dict(), allow_nan=False)
        self.assertEqual(r.to_dict(), outline.match_partial(self.target, matcher=self.matcher).to_dict())
        color = np.zeros((210,220,3), np.uint8)
        np.testing.assert_array_equal(r.render_debug(color), r.render_debug(color[..., ::-1], color_order='BGR')[..., ::-1])
        np.testing.assert_array_equal(before, self.current)
        detached = r.to_dict(); detached['best']['scale_x'] = 99
        self.assertNotEqual(r.best['scale_x'], 99)

    def test_invalid_parameters_and_inputs(self):
        for kwargs in [dict(scale_range=(0,2)), dict(axis_ratio_range=(1,2)),
                       dict(edge_samples=True), dict(max_hypotheses=0), dict(corner_turn_deg=180),
                       dict(axis_angle_deg=np.nan), dict(max_edge_rms=0), dict(simplify_px=-1),
                       dict(min_corner_condition=1), dict(ambiguity_scale_ratio=1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CornerOutlineMatcher(**kwargs)
        with self.assertRaises(ValueError):
            self.matcher.match_contour(self.target, self.current, articulation={})
        with self.assertRaises(ValueError):
            self.matcher.match_contour(self.target, self.current, observed_edges=np.ones(len(self.current)))


if __name__ == '__main__':
    unittest.main()
