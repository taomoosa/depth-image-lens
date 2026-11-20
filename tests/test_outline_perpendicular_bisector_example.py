"""Cross-sections of an exterior contour through an input segment's midpoint."""
import importlib.util
import json
import unittest

import numpy as np

from examples.example_outline_perpendicular_bisector import (
    largest_contour_bisector,
    perpendicular_bisector_intersections as intersect,
)


class BisectorGeometryTests(unittest.TestCase):
    def test_signed_distances_and_reversing_endpoints(self):
        rectangle = [(0, 0), (8, 0), (8, 6), (0, 6)]
        row = intersect(rectangle, (3, 2), (5, 2))
        self.assertEqual(row['midpoint_xy'], [4, 2])
        self.assertEqual(row['bisector_direction_xy'], [0, 1])
        self.assertEqual(row['intersections'], [
            dict(xy=[4, 0], signed_distance_px=-2, distance_px=2),
            dict(xy=[4, 6], signed_distance_px=4, distance_px=4),
        ])
        reverse = intersect(rectangle, (5, 2), (3, 2))
        forward = {tuple(hit['xy']): hit for hit in row['intersections']}
        for hit in reverse['intersections']:
            old = forward[tuple(hit['xy'])]
            self.assertEqual(hit['distance_px'], old['distance_px'])
            self.assertEqual(hit['signed_distance_px'], -old['signed_distance_px'])

    def test_oblique_interpolation_includes_the_closing_edge(self):
        rectangle = [(0, 0), (6, 0), (6, 4), (0, 4)]
        # The bisector is 5x + 2y = 2.9; its second hit is on the last-to-first edge.
        row = intersect(rectangle, (-2.32, -.928), (3.32, 1.328))
        hits = row['intersections']
        np.testing.assert_allclose([hit['xy'] for hit in hits], [[.58, 0], [0, 1.45]], atol=1e-9)
        for hit in hits:
            self.assertAlmostEqual(np.linalg.norm(np.array(hit['xy']) - row['midpoint_xy']),
                                   hit['distance_px'])
        json.dumps(row, allow_nan=False)

    def test_each_shared_vertex_or_tangent_reported_once(self):
        diamond = [(0, 2), (2, 0), (4, 2), (2, 4)]
        through = intersect(diamond, (1, 2), (3, 2))
        self.assertEqual([h['xy'] for h in through['intersections']], [[2, 0], [2, 4]])
        tangent = intersect(diamond, (3, 2), (5, 2))
        self.assertEqual(tangent['intersections'], [dict(xy=[4, 2], signed_distance_px=0, distance_px=0)])

    def test_collinear_edges_are_merged_and_endpoints_not_repeated(self):
        ring = [(0, 0), (2, 0), (4, 0), (4, 4), (2, 4), (2, 1), (1, 0), (0, 4)]
        # The last notch touches the same interval; its vertex is not an extra point.
        row = intersect(ring, (2, -1), (2, 1))
        self.assertEqual(row['intersections'], [])
        self.assertEqual(row['overlaps'], [dict(
            start=dict(xy=[4, 0], signed_distance_px=-2, distance_px=2),
            end=dict(xy=[0, 0], signed_distance_px=2, distance_px=2),
        )])

    def test_overlap_and_an_independent_point(self):
        ring = [(0, 0), (4, 0), (4, 4), (6, 4), (6, -2), (0, -2)]
        row = intersect(ring, (2, -1), (2, 1))
        self.assertEqual([h['xy'] for h in row['intersections']], [[6, 0]])
        self.assertEqual(len(row['overlaps']), 1)

    def test_no_hit_empty_ring_and_invalid_endpoints(self):
        ring = [(0, 0), (4, 0), (4, 4), (0, 4)]
        for points in (ring, []):
            row = intersect(points, (6, 0), (8, 0))
            self.assertEqual(row['intersections'], [])
            self.assertEqual(row['overlaps'], [])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            intersect(ring, (1, 2), (1, 2))
        with self.assertRaisesRegex(ValueError, 'finite XY pairs'):
            intersect(ring, (np.inf, 2), (1, 2))


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV required')
class BisectorMaskExampleTests(unittest.TestCase):
    def test_largest_concave_contour_has_four_hits_hole_is_ignored(self):
        mask = np.zeros((150, 150), dtype=bool)
        # This small earlier component also intersects the line but is not selected.
        mask[5:56, 3:8] = True
        mask[20:121, 20:131] = True
        mask[20:86, 55:95] = False
        mask[38:61, 100:105] = False
        row = largest_contour_bisector(mask, (75, 30), (75, 70))
        self.assertEqual(row['status'], 'intersected')
        self.assertEqual(row['contour_id'], 1)
        self.assertEqual([h['xy'] for h in row['intersections']],
                         [[130, 50], [95, 50], [54, 50], [20, 50]])
        self.assertEqual([h['signed_distance_px'] for h in row['intersections']], [-55, -20, 21, 55])

    def test_no_visible_contour_and_selected_contour_missed(self):
        shape = np.zeros((30, 30), dtype=bool)
        a, b = (3, 1), (5, 1)
        self.assertEqual(largest_contour_bisector(shape, a, b)['status'], 'no_visible_contour')
        shape[10:20, 10:20] = True
        self.assertEqual(largest_contour_bisector(shape, a, b)['status'], 'no_intersection')


if __name__ == '__main__':
    unittest.main()
