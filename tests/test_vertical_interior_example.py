"""Column-only selection and exact visible component extraction examples."""
import importlib.util
import json
import unittest
from dataclasses import replace
from copy import deepcopy
import numpy as np
from examples.example_vertical_interior import select_vertical_interior, contour_component_mask


class VerticalInteriorTests(unittest.TestCase):
    def mask(self, runs, height=20):
        a = np.zeros((height, 5), bool)
        for start, end in runs:
            a[start:end+1, 2] = True
        return a

    def test_odd_even_and_singleton(self):
        for run, y, clearance in [((3, 9), 6, 3), ((3, 10), 6, 3), ((5, 5), 5, 0)]:
            with self.subTest(run=run):
                row = select_vertical_interior(self.mask([run]), 2)
                self.assertEqual(row['point_xy'], [2, y])
                self.assertEqual(row['vertical_clearance_px'], clearance)
                self.assertEqual(row['run_y_inclusive'], list(run))

    def test_equal_length_topmost_and_equal_clearance_longest(self):
        a = self.mask([(2, 6), (10, 14)])
        self.assertEqual(select_vertical_interior(a, 2)['run_y_inclusive'], [2, 6])
        a = self.mask([(2, 6), (10, 15)])
        self.assertEqual(select_vertical_interior(a, 2)['run_y_inclusive'], [10, 15])

    def test_hole_splits_runs(self):
        row = select_vertical_interior(self.mask([(2, 5), (7, 14)]), 2)
        self.assertEqual(row['runs_y_inclusive'], [[2, 5], [7, 14]])
        self.assertEqual(row['point_xy'], [2, 10])

    def test_empty_column(self):
        self.assertIsNone(select_vertical_interior(self.mask([(2, 8)]), 1))

    def test_vertical_truncation(self):
        for run, top, bottom in [((0, 19), True, True), ((0, 4), True, False),
                                 ((15, 19), False, True), ((2, 9), False, False)]:
            row = select_vertical_interior(self.mask([run]), 2)
            self.assertEqual((row['touches_top'], row['touches_bottom']), (top, bottom))
            self.assertEqual(row['truncated'], top or bottom)

    def test_no_horizontal_influence_even_at_side_edge(self):
        a = self.mask([(3, 12)])
        expected = select_vertical_interior(a, 2)
        a[:, [0, 1, 3, 4]] = True
        self.assertEqual(select_vertical_interior(a, 2), expected)
        edge = select_vertical_interior(a[:, 2:3], 0)
        self.assertFalse(edge['truncated'])
        self.assertEqual(edge['vertical_clearance_px'], expected['vertical_clearance_px'])

    def test_noncontiguous_no_mutation_and_json(self):
        a = self.mask([(2, 10)])
        view = a[::-1, ::-1]
        before = view.copy()
        row = select_vertical_interior(view, np.int64(2))
        np.testing.assert_array_equal(before, view)
        self.assertEqual(json.loads(json.dumps(row, allow_nan=False)), row)

    def test_invalid_inputs(self):
        for x in [-1, 5, 1.5, True, '2']:
            with self.subTest(x=x), self.assertRaises(ValueError):
                select_vertical_interior(self.mask([]), x)
        for a in [np.zeros((3, 4), np.uint8), np.zeros(3, bool), np.zeros((0, 4), bool)]:
            with self.assertRaises(ValueError):
                select_vertical_interior(a, 0)

    def test_exhaustive_short_columns_maximize_clearance(self):
        for bits in range(256):
            col = np.array([(bits >> y) & 1 for y in range(8)], bool)
            row = select_vertical_interior(col[:, None], 0)
            if not col.any():
                self.assertIsNone(row)
                continue
            clearances = []
            for y in np.flatnonzero(col):
                lo, hi = int(y), int(y)
                while lo > 0 and col[lo-1]: lo -= 1
                while hi < 7 and col[hi+1]: hi += 1
                clearances.append(min(y-lo, hi-y))
            self.assertEqual(row['vertical_clearance_px'], max(clearances))


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV required')
class ContourComponentTests(unittest.TestCase):
    def result(self):
        from depth_image_lens import OuterContourDetector
        mask = np.zeros((50, 70), bool)
        mask[2:45, 2:45] = True
        mask[10:35, 10:35] = False
        mask[18:25, 18:25] = True  # Separate object inside a hole.
        mask[8:22, 55:65] = True
        return OuterContourDetector().detect(mask)

    def test_holes_and_nested_island_preserved(self):
        result = self.result()
        before = result.object_mask.copy()
        selected = contour_component_mask(result)
        self.assertTrue(selected[3, 3])
        self.assertFalse(selected[12, 12])
        self.assertFalse(selected[20, 20])
        self.assertFalse(selected[10, 60])
        self.assertEqual(int(selected.sum()), result.largest_contour()['foreground_pixel_count'])
        self.assertEqual(select_vertical_interior(selected, 20)['runs_y_inclusive'], [[2, 9], [35, 44]])
        selected[:] = False
        np.testing.assert_array_equal(before, result.object_mask)

    def test_nonindex_ids_and_observed_mask(self):
        result = self.result()
        records = deepcopy(result.contours)
        for c in records: c['id'] += 40
        modified = replace(result, contours=records)
        for contour in records:
            selected = contour_component_mask(modified, contour['id'])
            self.assertEqual(int(selected.sum()), contour['foreground_pixel_count'])
        from depth_image_lens import OuterContourDetector
        mask = np.ones((20, 30), bool)
        observed = np.ones_like(mask); observed[:, 14:16] = False
        result = OuterContourDetector().detect(mask, observed_mask=observed)
        selected = contour_component_mask(result)
        self.assertEqual(int(selected.sum()), 280)
        self.assertFalse(selected[:, 14:].any())

    def test_empty_and_unknown_id(self):
        from depth_image_lens import OuterContourDetector
        result = OuterContourDetector().detect(np.zeros((8, 9), bool))
        self.assertFalse(contour_component_mask(result).any())
        for value in [0, True, 0.5]:
            with self.assertRaises(ValueError): contour_component_mask(result, value)

    def test_malformed_contour_start_rejected(self):
        result = self.result()
        original = result.largest_contour()
        cases = [[], None, [[-1, 15]], [[15, -1]], [[70, 15]], [[15, 50]],
                 [[1]], [[1, 2, 3]], [None], [1], [[1.0, 2]], [[1, np.nan]],
                 [[np.inf, 2]], [[True, 2]], [['1', 2]], [[[1], [2]]]]
        for points in cases:
            with self.subTest(points=points):
                contour = deepcopy(original)
                contour['points_xy'] = points
                malformed = replace(result, contours=[contour])
                with self.assertRaises(ValueError):
                    contour_component_mask(malformed, contour['id'])
        contour = deepcopy(original)
        del contour['points_xy']
        with self.assertRaises(ValueError):
            contour_component_mask(replace(result, contours=[contour]), contour['id'])
        # Reproduce negative indexing into an unrelated right-edge component.
        from depth_image_lens import OuterContourDetector
        mask = np.zeros((60, 90), bool)
        mask[2:50, 2:50] = True; mask[10:40, 10:40] = False
        mask[20:28, 20:28] = True; mask[8:30, 75:90] = True
        result = OuterContourDetector().detect(mask)
        contour = result.largest_contour()
        contour['points_xy'][0] = [-1, 15]
        with self.assertRaises(ValueError):
            contour_component_mask(replace(result, contours=[contour]), contour['id'])

    def test_boundary_result_input(self):
        from depth_image_lens import OverheadBoundaryDetector
        mask = np.zeros((24, 32), bool)
        mask[2:9, 4:28] = True; mask[12:22, 8:24] = True
        result = OverheadBoundaryDetector().detect(
            np.full((24, 32, 3), 200, np.uint8), color_order='RGB', object_mask=mask)
        self.assertEqual(select_vertical_interior(result.object_mask, 16)['point_xy'], [16, 16])


if __name__ == '__main__':
    unittest.main()
