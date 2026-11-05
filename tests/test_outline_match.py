"""Synthetic geometry contracts, not real garment classification accuracy."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from depth_image_lens import (OuterContourDetector, PartialOutlineMatcher,
                              OutlineArticulation)


def shirt():
    # Hem -> armpit -> two cuff corners -> shoulder -> neckline.
    target = np.array([[60,155], [60,70], [30,85], [20,60], [65,30], [90,35]], float)
    rest = np.array([[110,35], [135,30], [180,60], [170,85], [140,70], [140,155]], float)
    return target, np.vstack([target, rest])


def sleeve(points, angle):
    q = points.copy()
    radians = np.radians(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
    q[2:4] = (q[2:4] - [62.5, 50]) @ rotation.T + [62.5, 50]
    return q


def mask_of(points, shape=(210,220)):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask > 0


class PartialOutlineTests(unittest.TestCase):
    def setUp(self):
        self.target, self.current = shirt()
        self.matcher = PartialOutlineMatcher()
        self.art = OutlineArticulation(2, 3, (62.5, 50))

    def test_similarity_seam_and_reverse(self):
        for scale, angle, reverse, roll in [(1., 0., False, 0), (.55, 63., True, 4),
                                             (2.3, -133., False, 9)]:
            with self.subTest(scale=scale, angle=angle):
                r = np.radians(angle)
                rotation = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
                current = scale*self.current @ rotation.T + [330, -65]
                current = np.roll(current[::-1] if reverse else current, roll, axis=0)
                result = self.matcher.match_contour(self.target, current)
                self.assertEqual(result.status, 'matched')
                self.assertLess(abs(result.best['scale']/scale-1), .025)
                expected = scale*self.target[[0, -1]] @ rotation.T + [330, -65]
                np.testing.assert_allclose([result.best['start_xy'], result.best['end_xy']], expected, atol=scale*2)
                self.assertLess(result.best['score'], .2)

    def test_reversed_template(self):
        r = self.matcher.match_contour(self.target[::-1], self.current)
        self.assertTrue(r.matched)
        np.testing.assert_allclose(r.best['start_xy'], self.target[-1], atol=2)

    def test_raster_rotation_scale_staircase_bias(self):
        for scale, angle in [(.5,36), (.8,-71), (1.4,109)]:
            theta = np.radians(angle)
            rotation = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
            current = scale*(self.current-[100,100])@rotation.T+[180,180]
            outline = OuterContourDetector().detect(mask_of(current,(360,360)))
            r = outline.match_partial(self.target, articulation=self.art)
            self.assertEqual(r.status, 'matched')
            self.assertLess(r.best['score'], .8)

    def test_ambiguity_on_one_contour(self):
        angles = np.linspace(0,2*np.pi,180,endpoint=False)
        current = np.c_[100+80*np.cos(angles),100+80*np.sin(angles)]
        result = self.matcher.match_contour(current[:64],current)
        self.assertEqual(result.status, 'ambiguous')
        self.assertGreaterEqual(len([c for c in result.candidates if c['passed']]), 2)

    def test_moderate_sleeve_rotations_on_raster_masks(self):
        for angle in (-25, -13, 17, 28):
            with self.subTest(angle=angle):
                outline = OuterContourDetector().detect(mask_of(sleeve(self.current, angle)))
                r = outline.match_partial(self.target, articulation=self.art)
                self.assertEqual(r.status, 'matched')
                self.assertLessEqual(abs(r.best['articulation_angle_deg']), 30)
                self.assertTrue(r.best['supported'])

    def test_articulation_is_opt_in_and_strict_rigid_control(self):
        current = sleeve(self.current, 30)
        rigid = PartialOutlineMatcher(max_rms=.008, max_fine_turn_deg=8)
        self.assertFalse(rigid.match_contour(self.target, current).matched)
        self.assertTrue(rigid.match_contour(self.target, current, articulation=self.art).matched)

    def curved_reference(self, amplitude):
        u = np.linspace(0, 1, 85)
        side = self.current[0]*(1-u[:, None]) + self.current[1]*u[:, None]
        side[:, 0] += amplitude*np.sin(np.pi*u)
        outline = OuterContourDetector(corner_turn_deg=35).detect(
            mask_of(np.vstack([side[:-1], self.current[1:]])))
        contour = outline.largest_contour()
        points = np.asarray(contour['points_xy'], float)
        first, last = [min(contour['corners'], key=lambda c: np.linalg.norm(np.asarray(c['xy'])-p))
                       for p in ([60, 155], [66, 30])]
        start = first['point_index']
        count = (last['point_index']-start) % len(points)
        corners = sorted([c for c in contour['corners']
                          if (c['point_index']-start) % len(points) <= count],
                         key=lambda c: (c['point_index']-start) % len(points))
        sparse = np.asarray([c['xy'] for c in corners], float)
        dense = points[(start+np.arange(count+1)) % len(points)]
        return outline, sparse, dense

    def test_articulation_preserves_rigid_search_at_acceptance_threshold(self):
        outline, sparse, _ = self.curved_reference(8)
        # The old joint seed suppression lost the rigid fit: score .977 became
        # 1.060, scale 1.088 and sleeve angle -10, incorrectly failing the match.
        for factor in (1., .9):
            limits = {k: v*factor for k, v in self.matcher.parameters.items()
                      if k.startswith('max_') and k != 'max_candidates'}
            matcher = PartialOutlineMatcher(**limits)
            rigid = outline.match_partial(sparse, matcher=matcher)
            flexible = outline.match_partial(sparse, matcher=matcher, articulation=self.art)
            self.assertEqual(rigid.status, 'matched')
            self.assertEqual(flexible.status, 'matched')
            self.assertLessEqual(flexible.best['score'], rigid.best['score']+1e-12)
            self.assertTrue(flexible.diagnostics['rigid_baseline_preserved'])

    def test_same_mask_dense_arc_preserves_curve_lost_by_corner_chords(self):
        outline, sparse, dense = self.curved_reference(16)
        coarse = outline.match_partial(sparse)
        original = outline.match_partial(dense)
        self.assertEqual(coarse.status, 'not_matched')
        self.assertEqual(original.status, 'matched')
        self.assertLess(original.best['score'], .1)
        self.assertAlmostEqual(original.best['scale'], 1., delta=.01)

    def test_noncorner_endpoints_across_seam_without_detected_corners(self):
        outline = OuterContourDetector(corner_turn_deg=180).detect(mask_of(self.current))
        contour = outline.largest_contour()
        self.assertEqual(contour['corners'], [])
        points = np.asarray(contour['points_xy'], float)
        start, end = [int(np.argmin(np.linalg.norm(points-p, axis=1)))
                      for p in ([60, 120], [100, 35])]
        self.assertGreater(start, end)  # wraps the array seam
        target = points[(start+np.arange((end-start) % len(points)+1)) % len(points)]
        result = outline.match_partial(target)
        self.assertEqual(result.status, 'matched')
        np.testing.assert_allclose([result.best['start_xy'], result.best['end_xy']],
                                   target[[0, -1]], atol=2)

    def test_fold_and_missing_sleeve_negatives(self):
        cases = [sleeve(self.current, 90), sleeve(self.current, -75),
                 np.delete(self.current, [2, 3], axis=0)]
        for points in cases:
            outline = OuterContourDetector().detect(mask_of(points))
            r = outline.match_partial(self.target, articulation=self.art)
            self.assertEqual(r.status, 'not_matched')
            self.assertGreater(r.best['score'], 1)

    def test_unrelated_silhouettes_and_crumpled_shape(self):
        angles = np.linspace(0, 2*np.pi, 160, endpoint=False)
        cases = [np.array([[20,20], [190,20], [190,180], [20,180]]),
                 np.c_[100+70*np.cos(angles), 100+60*np.sin(angles)],
                 np.c_[100+(50+12*np.sin(13*angles))*np.cos(angles),
                       100+(50+12*np.sin(13*angles))*np.sin(angles)]]
        for current in cases:
            r = self.matcher.match_contour(self.target, current, articulation=self.art)
            self.assertEqual(r.status, 'not_matched')

    def test_local_zigzag_is_not_hidden_by_global_fit(self):
        # Add folds on the long torso edge, retaining the original cuff/shoulder.
        t = np.linspace(0, 1, 61)
        edge = self.current[0]*(1-t[:,None]) + self.current[1]*t[:,None]
        edge[:,0] += 6*np.sin(t*14*np.pi)*np.sin(t*np.pi)**2
        current = np.vstack([edge[:-1], self.current[1:]])
        r = self.matcher.match_contour(self.target, current, articulation=self.art)
        self.assertEqual(r.status, 'not_matched')

    def test_small_sensor_noise(self):
        rng = np.random.default_rng(814)
        pieces = []
        for start, end in zip(self.current, np.roll(self.current, -1, axis=0)):
            u = np.linspace(0, 1, 30, endpoint=False)
            pieces.append(start[None,:]*(1-u[:,None])+end[None,:]*u[:,None])
        current = np.vstack(pieces) + rng.normal(0, .13, (360, 2))
        r = self.matcher.match_contour(self.target, current, articulation=self.art)
        self.assertEqual(r.status, 'matched')

    def test_missing_edge_inside_target_withholds_match(self):
        observed = np.ones(len(self.current), bool)
        observed[2] = False
        r = self.matcher.match_contour(self.target, self.current, observed_edges=observed)
        self.assertEqual(r.status, 'unknown')
        self.assertIsNone(r.matched)
        self.assertFalse(r.best['supported'])

    def test_unobserved_opposite_side_does_not_erase_visible_match(self):
        observed = np.ones(len(self.current), bool)
        observed[8] = False
        r = self.matcher.match_contour(self.target, self.current, observed_edges=observed)
        self.assertEqual(r.status, 'matched')

    def test_one_short_missing_edge_between_samples(self):
        # The unavailable interval is far shorter than the matching sample spacing.
        c = np.vstack([self.current[:3], [[25.01,72.525], [25,72.5]], self.current[3:]])
        observed = np.ones(len(c), bool)
        observed[3] = False
        r = self.matcher.match_contour(self.target, c, observed_edges=observed)
        self.assertEqual(r.status, 'unknown')
        self.assertFalse(r.best['supported'])

    def test_observation_domain_artificial_edge(self):
        mask = mask_of(self.current)
        observed = np.ones(mask.shape, bool)
        observed[:, :32] = False
        outline = OuterContourDetector().detect(mask, observed_mask=observed)
        r = outline.match_partial(self.target, articulation=self.art)
        self.assertEqual(r.status, 'unknown')

    def test_two_objects_are_ambiguous(self):
        mask = mask_of(self.current, (230,480)) | mask_of(self.current+[240,0], (230,480))
        r = OuterContourDetector().detect(mask).match_partial(self.target)
        self.assertEqual(r.status, 'ambiguous')
        self.assertIsNone(r.matched)
        self.assertTrue(r.diagnostics['candidate_found'])
        self.assertEqual(len({c['contour_id'] for c in r.candidates if c['passed']}), 2)

    def test_fully_observed_match_beats_better_occluded_lookalike(self):
        mask = mask_of(self.current, (230,480)) | mask_of(sleeve(self.current, 10)+[240,0], (230,480))
        observed = np.ones(mask.shape, bool)
        observed[69:75, 27:34] = False
        outline = OuterContourDetector().detect(mask, observed_mask=observed)
        r = outline.match_partial(self.target, articulation=self.art)
        self.assertEqual(r.status, 'matched')
        self.assertGreater(r.best['start_xy'][0], 240)

    def test_holes_are_never_target_arcs(self):
        mask = np.zeros((240,250), bool)
        mask[5:225,5:235] = True
        mask[mask_of(self.current, mask.shape)] = False
        outline = OuterContourDetector().detect(mask)
        self.assertTrue(outline.holes)
        self.assertEqual(outline.match_partial(self.target).status, 'not_matched')

    def test_straight_template_empty_tiny_and_crossed_contours(self):
        r = self.matcher.match_contour([[0,0], [25,0], [50,0]], self.current)
        self.assertEqual(r.status, 'unknown')
        self.assertIn('target_has_insufficient_turn_structure', r.diagnostics['reasons'])
        r = OuterContourDetector().detect(np.zeros((100,100),bool)).match_partial(self.target)
        self.assertEqual(r.status, 'unknown')
        r = self.matcher.match_contour(self.target, [[0,0],[1,0],[1,1],[0,1]])
        self.assertEqual(r.status, 'unknown')
        r = self.matcher.match_contour(self.target, [[0,0],[100,100],[0,100],[100,0]])
        self.assertEqual(r.status, 'unknown')

    def test_reflection_is_not_implicitly_allowed(self):
        # Remove the other shirt side to avoid an actual mirror counterpart.
        current = np.vstack([self.target, [[170,10],[180,180]]])
        result = self.matcher.match_contour(self.target, current * [-1,1])
        self.assertEqual(result.status, 'not_matched')

    def test_determinism_immutability_and_strict_json(self):
        before = self.current.copy()
        a = self.matcher.match_contour(self.target, self.current, articulation=self.art)
        b = self.matcher.match_contour(self.target, self.current, articulation=self.art)
        self.assertEqual(a.to_dict(), b.to_dict())
        json.dumps(a.to_dict(), allow_nan=False)
        np.testing.assert_array_equal(before, self.current)
        detached = a.to_dict()
        detached['best']['start_xy'][0] = -1000
        self.assertNotEqual(a.best['start_xy'][0], -1000)
        self.assertFalse(a.diagnostics['search_complete'])
        self.assertGreater(a.diagnostics['evaluated_hypotheses'], 0)

    def test_invalid_inputs_and_configuration(self):
        for bad in [[], [[0,0],[1,1]], [[0,0],[1,1],[0,0]], [[0,0],[1,1],[np.nan,2]],
                    [[0,0],[1,1],[1,1]], [['a','b'],['c','d'],['e','f']]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.matcher.match_contour(bad, self.current)
        for config in [dict(fraction_range=(.5,.2)),dict(fraction_range=(0,1)),
                       dict(search_samples=True),dict(sample_count=16),dict(max_rms=np.nan),
                       dict(max_coarse_turn_deg=181),dict(min_arc_length_px=0),dict(simplify_px=-1)]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                PartialOutlineMatcher(**config)
        with self.assertRaises(ValueError):
            self.matcher.match_contour(self.target, self.current, observed_edges=np.ones(12))
        with self.assertRaises(ValueError):
            self.matcher.match_contour(self.target, self.current, articulation={})
        for art in [OutlineArticulation(0,5,(0,0)),OutlineArticulation(-1,2,(0,0)),
                    OutlineArticulation(2,3,(0,np.nan)),OutlineArticulation(2,3,(0,0),max_angle_deg=90)]:
            with self.subTest(art=art), self.assertRaises(ValueError):
                self.matcher.match_contour(self.target, self.current, articulation=art)

    def test_debug_rgb_bgr_and_no_overwrite(self):
        result = self.matcher.match_contour(self.target, self.current)
        color = np.zeros((220,220,3), np.uint8)
        color[...,0] = 65
        before = color.copy()
        rgb = result.render_debug(color)
        bgr = result.render_debug(color[...,::-1], color_order='BGR')
        np.testing.assert_array_equal(rgb, bgr[...,::-1])
        np.testing.assert_array_equal(color, before)
        self.assertTrue(np.any(rgb != color))
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)/'match.png'
            result.save_debug(dest,color)
            self.assertTrue(dest.is_file())
            with self.assertRaises(FileExistsError):
                result.save_debug(dest,color)
        with self.assertRaises(ValueError):
            result.render_debug(color, candidate_index=999)


if __name__ == '__main__':
    unittest.main()
