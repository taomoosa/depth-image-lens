"""Inward corner offsets are separate, conservative pixel-geometry queries."""
from copy import deepcopy
import json
import unittest

import cv2
import numpy as np

from depth_image_lens import CameraIntrinsics, OuterContourDetector


def rectangle():
    mask = np.zeros((150, 190), bool)
    mask[30:121, 35:156] = True
    return mask


class CornerInteriorTests(unittest.TestCase):
    def test_rectangle_exact_displacement_and_detached_records(self):
        result = OuterContourDetector().detect(rectangle())
        before = deepcopy(result.to_dict())
        rows = result.corner_interior_points(10)
        self.assertEqual(len(rows), 4)
        for row, corner, signs in zip(rows, result.contours[0]['corners'],
                                     [(1, 1), (-1, 1), (-1, -1), (1, -1)]):
            self.assertEqual(row['status'], 'computed')
            self.assertIsNone(row['reason'])
            self.assertEqual(row['corner_id'], corner['id'])
            self.assertEqual(row['source_xy'], corner['xy'])
            delta = 10 * np.array(signs) / np.sqrt(2)
            np.testing.assert_allclose(row['xy'], np.array(corner['xy']) + delta)
            self.assertAlmostEqual(np.linalg.norm(np.array(row['xy']) - row['source_xy']), 10)
            self.assertEqual(row['sample_xy'], np.floor(np.array(row['xy']) + .5).astype(int).tolist())
        json.dumps(rows, allow_nan=False)
        rows[0]['xy'][0] = rows[0]['source_xy'][0] = -100
        self.assertEqual(result.to_dict(), before)

    def test_zero_fractional_and_rounding(self):
        result = OuterContourDetector().detect(rectangle())
        for row in result.corner_interior_points(0):
            self.assertEqual(row['status'], 'computed')
            self.assertEqual(row['xy'], row['source_xy'])
            self.assertEqual(row['sample_xy'], row['source_xy'])
        row = result.corner_interior_points(.25)[0]
        self.assertEqual(row['status'], 'computed')
        self.assertAlmostEqual(np.linalg.norm(np.array(row['xy']) - row['source_xy']), .25)
        self.assertEqual(row['sample_xy'], row['source_xy'])
        # Half-up, not NumPy's ties-to-even: the top-left Y center is even (30).
        row = result.corner_interior_points(np.sqrt(.5))[0]
        self.assertEqual(row['sample_xy'], [36, 31])
        for offset in [1e-6, 1e-8]:
            self.assertTrue(all(r['status'] == 'computed' for r in result.corner_interior_points(offset)))

    def test_concave_corner_moves_into_material_not_notch(self):
        mask = rectangle()
        mask[70:121, 95:156] = False
        result = OuterContourDetector().detect(mask)
        rows = {r['corner_id']: r for r in result.corner_interior_points(10)}
        concave = [c for c in result.contours[0]['corners'] if c['kind'] == 'concave']
        self.assertEqual(len(concave), 1)
        row = rows[concave[0]['id']]
        self.assertEqual(row['status'], 'computed')
        self.assertTrue(np.all(np.array(row['xy']) < row['source_xy']))
        x, y = row['sample_xy']
        self.assertTrue(mask[y, x])

    def test_rotation_translation_and_mirror(self):
        mask = np.zeros((220, 250), np.uint8)
        cv2.fillPoly(mask, [np.array([[60, 30], [210, 90], [170, 190], [20, 130]])], 1)
        detector = OuterContourDetector()
        result = detector.detect(mask > 0)
        rows = result.corner_interior_points(7.5)
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual(row['status'], 'computed')
            self.assertAlmostEqual(np.linalg.norm(np.array(row['xy']) - row['source_xy']), 7.5)
        translated = detector.detect(np.pad(mask, ((11, 0), (9, 0))) > 0).corner_interior_points(7.5)
        np.testing.assert_allclose(np.array([r['xy'] for r in translated]) - [9, 11],
                                   [r['xy'] for r in rows])
        # A 90 degree rotation and reflection retain the inward direction. On
        # this axis-aligned fixture the detected raw corner centers are exact.
        original = detector.detect(rectangle()).corner_interior_points(8)
        for transformed, transform in [
                (np.rot90(rectangle()), lambda xy: [xy[1], 189 - xy[0]]),
                (rectangle()[:, ::-1], lambda xy: [189 - xy[0], xy[1]])]:
            expected = {tuple(transform(r['source_xy'])): transform(r['xy']) for r in original}
            for row in detector.detect(transformed).corner_interior_points(8):
                np.testing.assert_allclose(row['xy'], expected[tuple(row['source_xy'])])

    def test_component_selection_and_nonconsecutive_ids(self):
        mask = np.zeros((180, 280), bool)
        mask[20:80, 20:80] = True
        mask[30:150, 130:260] = True
        result = OuterContourDetector().detect(mask)
        result.contours[0]['id'], result.contours[1]['id'] = 8, 37
        for corner in result.contours[1]['corners']:
            corner['id'] += 100
        self.assertTrue(all(r['contour_id'] == 37 for r in result.corner_interior_points(5)))
        self.assertTrue(all(r['corner_id'] >= 100 for r in result.corner_interior_points(5)))
        self.assertTrue(all(r['contour_id'] == 8 for r in result.corner_interior_points(5, 8)))
        self.assertEqual(result.corner_interior_points(5), result.corner_interior_points(5, np.int64(37)))

    def test_empty_tiny_and_smooth_contours(self):
        detector = OuterContourDetector()
        mask = np.zeros((160, 160), bool)
        self.assertEqual(detector.detect(mask).corner_interior_points(5), [])
        mask[20, 20] = True
        self.assertEqual(detector.detect(mask).corner_interior_points(5), [])
        y, x = np.indices(mask.shape)
        self.assertEqual(detector.detect((x - 80)**2 + (y - 80)**2 <= 50**2).corner_interior_points(5), [])

    def test_invalid_distance_and_ids(self):
        result = OuterContourDetector().detect(rectangle())
        for offset in [-1, np.inf, -np.inf, np.nan, True, np.bool_(False), '5', None,
                       [5], np.array(5), 1j, 10**400, np.longdouble('1e400')]:
            with self.subTest(offset=str(offset)), self.assertRaises(ValueError):
                result.corner_interior_points(offset)
        for contour_id in [True, '0', 0., -1, 999]:
            with self.subTest(contour_id=contour_id), self.assertRaises(ValueError):
                result.corner_interior_points(5, contour_id)
        with self.assertRaises(ValueError):
            OuterContourDetector().detect(np.zeros((10, 10), bool)).corner_interior_points(5, 0)
        duplicate = deepcopy(result)
        duplicate.contours[0]['corners'].append(deepcopy(duplicate.contours[0]['corners'][0]))
        with self.assertRaises(ValueError):
            duplicate.corner_interior_points(5)
        for offset in [np.int64(5), np.float32(5)]:
            self.assertEqual(result.corner_interior_points(offset), result.corner_interior_points(5))

    def test_large_offset_is_unavailable_never_clamped(self):
        result = OuterContourDetector().detect(rectangle())
        for offset in [300., np.finfo(float).max]:
            for row in result.corner_interior_points(offset):
                self.assertEqual(row['status'], 'unavailable')
                self.assertEqual(row['reason'], 'offset_point_not_inside_component')
                self.assertIsNone(row['xy'])
                self.assertIsNone(row['sample_xy'])

    def test_frame_and_observation_boundary_corners_unavailable(self):
        result = OuterContourDetector().detect(np.ones((100, 120), bool))
        rows = result.corner_interior_points(5)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['reason'] == 'corner_not_observed' for r in rows))
        observed = np.ones_like(rectangle())
        observed[:, 110:] = False
        result = OuterContourDetector().detect(rectangle(), observed_mask=observed)
        rows = result.corner_interior_points(5)
        self.assertEqual(sum(r['status'] == 'computed' for r in rows), 2)
        self.assertEqual(sum(r['reason'] == 'corner_not_observed' for r in rows), 2)

    def test_observation_gap_in_tangent_support(self):
        # The corner itself is observed, but one arm's support is not reliable.
        observed = np.ones_like(rectangle())
        observed[29, 41] = False
        result = OuterContourDetector().detect(rectangle(), observed_mask=observed)
        self.assertEqual(result.contours[0]['corners'][0]['visibility'], 'observed')
        self.assertEqual(result.corner_interior_points(5)[0]['reason'], 'unobserved_corner_support')

    def test_thin_region_and_opposite_boundary(self):
        mask = np.zeros((70, 150), bool)
        mask[20:25, 20:130] = True
        result = OuterContourDetector(corner_window_px=2, min_corner_spacing_px=2, smoothing_px=0).detect(mask)
        rows = result.corner_interior_points(2)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['status'] == 'computed' for r in rows))
        for offset in [4 * np.sqrt(2), 10]:
            self.assertTrue(all(r['status'] == 'unavailable' for r in result.corner_interior_points(offset)))

    def test_hole_endpoint_and_jump_across_hole(self):
        mask = np.zeros((180, 180), bool)
        mask[20:161, 20:161] = True
        mask[50:80, 50:80] = False
        result = OuterContourDetector().detect(mask)
        self.assertEqual(result.corner_interior_points(50)[0]['reason'], 'offset_point_not_inside_component')
        row = result.corner_interior_points(110)[0]
        self.assertEqual(row['reason'], 'offset_path_touches_boundary')
        self.assertIsNone(row['xy'])
        self.assertEqual(len(result.corner_interior_points(5)), 4)  # no hole corners

    def test_unobserved_interior_hole_blocks_path(self):
        mask = np.zeros((180, 180), bool)
        mask[20:161, 20:161] = True
        observed = np.ones_like(mask)
        observed[50:80, 50:80] = False
        result = OuterContourDetector().detect(mask, observed_mask=observed)
        self.assertEqual(result.corner_interior_points(110)[0]['reason'], 'offset_path_touches_boundary')

    def test_cannot_jump_into_separate_island_inside_hole(self):
        mask = np.zeros((200, 200), bool)
        mask[20:181, 20:181] = True
        mask[40:160, 40:160] = False
        mask[70:130, 70:130] = True
        result = OuterContourDetector().detect(mask)
        row = result.corner_interior_points(100)[0]
        self.assertEqual(row['reason'], 'offset_point_not_inside_component')
        self.assertEqual(row['contour_id'], result.largest_contour()['id'])

    def test_cannot_jump_across_notch_to_same_component(self):
        mask = np.zeros((180, 180), bool)
        mask[20:161, 20:161] = True
        mask[50:80, 20:100] = False
        result = OuterContourDetector().detect(mask)
        self.assertEqual(result.corner_interior_points(110)[0]['reason'], 'offset_path_touches_boundary')

    def test_self_touch_and_narrow_connection_are_unavailable(self):
        mask = np.zeros((120, 120), bool)
        mask[10:51, 10:51] = True
        mask[51:92, 51:92] = True
        result = OuterContourDetector().detect(mask)
        self.assertEqual(result.contours[0]['status'], 'ambiguous_self_touch')
        rows = result.corner_interior_points(5)
        self.assertTrue(rows)
        self.assertTrue(all(r['reason'] == 'ambiguous_or_insufficient_contour' for r in rows))

    def test_no_dependency_on_diagnostics_depth_or_input_layout(self):
        mask = rectangle()
        detector = OuterContourDetector()
        expected = detector.detect(mask).corner_interior_points(8)
        before = mask.copy()
        for options in [dict(diagnostics=True), dict(depth=np.zeros(mask.shape)),
                        dict(depth=np.ones(mask.shape), depth_scale_m=1.,
                             intrinsics=CameraIntrinsics(400, 400, 90, 75), registered_to_mask=True)]:
            self.assertEqual(detector.detect(mask, **options).corner_interior_points(8), expected)
        np.testing.assert_array_equal(mask, before)
        readonly = mask.copy()
        readonly.flags.writeable = False
        self.assertEqual(detector.detect(readonly).corner_interior_points(8), expected)
        self.assertEqual(detector.detect(np.asfortranarray(mask)).corner_interior_points(8), expected)


if __name__ == '__main__':
    unittest.main()
