"""Limited current-only skips keep original edge geometry and evidence intact."""
import json
import unittest
from unittest.mock import patch
import numpy as np

from depth_image_lens import CornerOutlineMatcher, OutlineArticulation, OuterContourDetector
from depth_image_lens.corner_outline_match import _correspondence_plans
from .test_outline_match import shirt, mask_of, sleeve


def bump(points, edge_index, amplitude):
    delta = points[edge_index+1]-points[edge_index]
    normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
    return np.insert(points, edge_index+1,
                     (points[edge_index]+points[edge_index+1])/2+amplitude*normal, axis=0)


class CornerSkipTests(unittest.TestCase):
    def setUp(self):
        self.target, self.current = shirt()
        self.one = bump(self.current, 1, 3.2)
        self.art = OutlineArticulation(2, 3, (62.5, 50))

    def match(self, current, skip, **kwargs):
        return CornerOutlineMatcher(max_skipped_corners=skip).match_contour(self.target, current, **kwargs)

    def test_one_extra_corner_passes_only_with_skip_and_edge_error_remains(self):
        self.assertEqual(self.match(self.one, 0).status, 'not_matched')
        result = self.match(self.one, 1)
        self.assertEqual(result.status, 'matched')
        self.assertEqual(result.best['skipped_corner_indices'], [2])
        self.assertEqual(result.best['skipped_corner_count'], 1)
        np.testing.assert_allclose(result.best['skipped_corner_xy'], self.one[[2]])
        self.assertLess(result.best['metrics']['corner_rms'], 1e-10)
        self.assertGreater(result.best['edges'][1]['rms_px'], 1.)
        self.assertGreater(result.best['edges'][1]['score'], .8)
        self.assertLessEqual(result.best['edges'][1]['score'], 1.)

    def test_two_extra_corners_on_separate_edges_share_one_budget(self):
        current = bump(bump(self.current, 2, 2.6), 1, 3.2)
        self.assertEqual(self.match(current, 1).status, 'not_matched')
        r = self.match(current, 2)
        self.assertEqual(r.status, 'matched')
        self.assertEqual(r.best['skipped_corner_count'], 2)

    def test_skip_and_sleeve_rotation_work_together(self):
        current = bump(sleeve(self.current, 20), 1, 3.8)
        self.assertEqual(self.match(current, 0, articulation=self.art).status, 'not_matched')
        r = self.match(current, 1, articulation=self.art)
        self.assertEqual(r.status, 'matched')
        self.assertEqual(r.best['articulation_angle_deg'], 20.)
        self.assertEqual(r.best['skipped_corner_count'], 1)

    def test_two_extra_corners_on_same_edge(self):
        v = self.current[2]-self.current[1]
        n = np.array([-v[1], v[0]])/np.linalg.norm(v)
        current = np.vstack([self.current[:2], self.current[1]+v/3+2*n,
                             self.current[1]+2*v/3-2*n, self.current[2:]])
        self.assertEqual(self.match(current, 1).status, 'not_matched')
        r = self.match(current, 2)
        self.assertEqual(r.status, 'matched')
        self.assertEqual(r.best['skipped_corner_indices'], [2, 3])
        self.assertGreater(r.best['edges'][1]['metrics']['turn_deg'], 5.)
        self.assertGreater(r.best['edges'][1]['metrics']['excess_turn_deg'], 50.)

    def test_endpoint_tail_skips_use_the_same_total_budget(self):
        current = bump(bump(self.current, 4, 2.6), 0, 8.)
        self.assertEqual(self.match(current, 1).status, 'not_matched')
        r = self.match(current, 2)
        self.assertEqual(r.status, 'matched')
        self.assertEqual(r.best['skipped_corner_count'], 2)
        self.assertGreater(r.best['edges'][0]['score'], .8)
        self.assertGreater(r.best['edges'][-1]['score'], .8)
        np.testing.assert_allclose([r.best['start_xy'], r.best['end_xy']], self.target[[0, -1]], atol=1e-8)

    def test_order_seam_reverse_and_indices(self):
        for q in (np.roll(self.one, 4, axis=0), np.roll(self.one[::-1], 4, axis=0)):
            r = self.match(q, 1)
            self.assertEqual(r.status, 'matched')
            b = r.best
            np.testing.assert_allclose(q[b['observed_corner_indices']], b['observed_corner_xy'], atol=1e-8)
            np.testing.assert_allclose(q[b['skipped_corner_indices']], b['skipped_corner_xy'], atol=1e-8)
            self.assertFalse(set(b['observed_corner_indices']) & set(b['skipped_corner_indices']))

    def test_large_fold_stays_in_edge_validation(self):
        current = bump(self.current, 1, 12.)
        for skip in (1, 2):
            r = self.match(current, skip)
            self.assertEqual(r.status, 'not_matched')
            self.assertEqual(r.best['skipped_corner_count'], 1)
            self.assertLess(r.best['metrics']['corner_rms'], 1e-10)
            self.assertGreater(r.best['edges'][1]['score'], 3.)

    def test_missing_target_bend_is_never_skipped(self):
        target = bump(self.target, 1, 3.2)
        r = CornerOutlineMatcher(max_skipped_corners=2).match_contour(target, self.current)
        self.assertEqual(r.status, 'not_matched')
        for candidate in r.candidates:
            self.assertEqual(len(candidate['observed_corner_indices']), r.diagnostics['target_corner_count'])

    def test_folds_and_disagreeing_extra_corners_remain_negative(self):
        cases = [sleeve(self.current, 90), np.delete(self.current, 2, axis=0),
                 np.vstack([self.current[:2], [[48,58], [40,75]], self.current[2:]])]
        for q in cases:
            r = self.match(q, 2, articulation=self.art)
            self.assertEqual(r.status, 'not_matched')

    def test_subsample_missing_edge_at_skipped_corner_is_not_bridged(self):
        near = self.one[2]+.001*(self.one[3]-self.one[2])
        current = np.insert(self.one, 3, near, axis=0)
        observed = np.ones(len(current), bool); observed[2] = False
        for q, good in [(current, observed),
                        (current[::-1], np.roll(observed[::-1], -1))]:
            r = self.match(q, 1, observed_edges=good)
            self.assertEqual(r.status, 'unknown')
            self.assertFalse(r.best['supported'])
        observed[:] = True; observed[-3] = False
        self.assertEqual(self.match(current, 1, observed_edges=observed).status, 'matched')

    def test_clipped_mask_and_two_matching_objects(self):
        # Larger raster fixture keeps pixel rounding below the edge limits.
        q = self.one*2
        mask = mask_of(q, (460,960)) | mask_of(q+[480,0], (460,960))
        matcher = CornerOutlineMatcher(max_skipped_corners=1)
        r = OuterContourDetector().detect(mask).match_partial(self.target*2, matcher=matcher)
        self.assertEqual(r.status, 'ambiguous')
        self.assertIn('multiple_distinct_visible_arcs_fit', r.diagnostics['reasons'])
        mask = mask_of(q, (460,480)); observed = np.ones_like(mask); observed[:, :64] = False
        r = OuterContourDetector().detect(mask, observed_mask=observed).match_partial(self.target*2, matcher=matcher)
        self.assertEqual(r.status, 'unknown')

    def test_budget_limits_withhold_partial_search_success(self):
        r = CornerOutlineMatcher(max_skipped_corners=1, max_hypotheses=500).match_contour(
            self.target, self.one, articulation=self.art)
        self.assertEqual(r.status, 'unknown')
        self.assertIn('search_budget_exceeded', r.diagnostics['reasons'])
        self.assertLessEqual(r.diagnostics['evaluated_hypotheses'], 500)
        self.assertTrue(r.diagnostics['candidate_found'])
        self.assertTrue(r.best['passed'])  # provisional fit is not a final match

    def test_budget_failure_survives_later_direction_and_object_results(self):
        import depth_image_lens.corner_outline_match as module
        mask = mask_of(self.current, (230,480)) | mask_of(self.current+[240,0], (230,480))
        outline = OuterContourDetector().detect(mask)
        original, calls = module._corners, [0]
        def corners(*args):
            calls[0] += 1
            if calls[0] == 2:
                return np.arange(30)  # first direction exceeds the search budget
            if calls[0] == 3:
                return np.arange(3)   # reverse direction has insufficient corners
            return original(*args)
        with patch.object(module, '_corners', side_effect=corners):
            r = CornerOutlineMatcher(max_skipped_corners=1, max_hypotheses=150).match(self.target, outline)
        self.assertTrue(r.diagnostics['candidate_found'])  # second object fits
        self.assertEqual(r.status, 'unknown')
        self.assertIn('search_budget_exceeded', r.diagnostics['reasons'])

    def test_plans_are_bounded_ordered_and_never_reuse_corners(self):
        for m in (3, 4, 6, 16):
            for count in (m, m+1, m+3):
                for budget in (0, 1, 2):
                    for offsets, tails in _correspondence_plans(m, count, budget):
                        self.assertEqual(len(offsets), m)
                        self.assertTrue(np.all(np.diff(offsets) > 0))
                        for before, after in tails:
                            used = offsets[-1]+1+before+after
                            self.assertLessEqual(used, count)
                            self.assertLessEqual(used-m, budget)
        plans = _correspondence_plans(4, 12, 2)
        self.assertEqual(sum(len(tails) for _, tails in plans), 21)

    def test_pruning_keeps_all_passing_hypotheses_and_ambiguity(self):
        # Suppress pruning by making the cheap bound look zero; the full candidate
        # calculation then restores the real ratios. Compare the actual outcome.
        matcher = CornerOutlineMatcher(max_skipped_corners=1)
        original = matcher._corner_quality
        original_candidate = matcher._candidate
        def cheap(*args):
            quality = original(*args)
            if quality is None:
                return None
            fitted, metrics, ratios = quality
            return fitted, metrics, dict.fromkeys(ratios, 0.)
        def full(target, ids, current, observed, interior, prev, nxt, fit, quality, angle, cid, reverse):
            real = original(target, ids, current, interior, fit, angle, self.art.max_angle_deg)
            return original_candidate(target, ids, current, observed, interior, prev, nxt, fit, real, angle, cid, reverse)
        pruned = matcher.match_contour(self.target, self.one, articulation=self.art)
        with patch.object(matcher, '_corner_quality', side_effect=cheap), patch.object(matcher, '_candidate', side_effect=full):
            unpruned = matcher.match_contour(self.target, self.one, articulation=self.art)
        self.assertEqual(pruned.status, unpruned.status)
        self.assertAlmostEqual(pruned.best['score'], unpruned.best['score'])
        self.assertGreater(pruned.diagnostics['corner_bound_pruned_hypotheses'], 0)
        self.assertEqual(unpruned.diagnostics['corner_bound_pruned_hypotheses'], 0)

    def test_zero_default_compatibility_and_json(self):
        default = CornerOutlineMatcher().match_contour(self.target, self.current, articulation=self.art)
        explicit = self.match(self.current, 0, articulation=self.art)
        self.assertEqual(default.to_dict(), explicit.to_dict())
        self.assertEqual(default.best['skipped_corner_count'], 0)
        r = self.match(self.one, 1)
        json.dumps(r.to_dict(), allow_nan=False)
        self.assertEqual(r.to_dict(), self.match(self.one, 1).to_dict())

    def test_skip_limit_validation(self):
        for value in (-1, 3, True, 1.5, None, np.int64(1)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CornerOutlineMatcher(max_skipped_corners=value)


if __name__ == '__main__':
    unittest.main()
