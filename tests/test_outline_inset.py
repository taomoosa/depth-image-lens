"""Whole visible component offsets never stitch or edit old corners."""
from copy import deepcopy
import json
from math import sqrt
import unittest

import numpy as np

from depth_image_lens import OuterContourDetector, OuterContourInsetResult


def blank():
    return np.zeros((52, 60), bool)


class OutlineInsetTests(unittest.TestCase):
    def setUp(self):
        self.detector = OuterContourDetector()

    def test_rectangle_exact_inset_in_input_grid_and_detached(self):
        mask = blank()
        mask[7:37, 13:53] = True
        result = self.detector.detect(mask)
        before = deepcopy(result.to_dict())
        inset = result.inset_component(4)
        self.assertIsInstance(inset, OuterContourInsetResult)
        self.assertEqual(inset.inset_mask.dtype, bool)
        expected = blank()
        expected[11:33, 17:49] = True
        np.testing.assert_array_equal(inset.inset_mask, expected)
        self.assertEqual(inset.summary['status'], 'computed')
        self.assertEqual(inset.summary['polygon_status'], 'simple')
        self.assertEqual(inset.summary['source_pixel_count'], 1200)
        self.assertEqual(inset.summary['inset_pixel_count'], 704)
        self.assertEqual(inset.components[0]['outer']['points_xy'][0], [17, 11])
        self.assertEqual(inset.components[0]['foreground_pixel_count'], 704)
        self.assertEqual(inset.components[0]['holes'], [])
        exported = inset.to_dict()
        json.dumps(exported, allow_nan=False)
        exported['components'][0]['outer']['points_xy'][0][0] = -1
        self.assertEqual(inset.components[0]['outer']['points_xy'][0][0], 17)
        inset.inset_mask[:] = False
        self.assertEqual(result.to_dict(), before)
        np.testing.assert_array_equal(result.object_mask, mask)

    def test_source_self_touch_does_not_gate_the_whole_inset(self):
        mask = blank()
        mask[5:24, 5:24] = True
        mask[24:43, 24:43] = True
        result = self.detector.detect(mask)
        self.assertEqual(result.contours[0]['status'], 'ambiguous_self_touch')
        self.assertTrue(all(r['status'] == 'unavailable' for r in result.corner_interior_points(2)))
        inset = result.inset_component(2)
        self.assertEqual(inset.summary['source_contour_status'], 'ambiguous_self_touch')
        self.assertEqual(inset.summary['status'], 'split')
        self.assertEqual(inset.summary['polygon_status'], 'simple')
        self.assertEqual([p['foreground_pixel_count'] for p in inset.components], [225, 225])
        expected = blank()
        expected[7:22, 7:22] = True
        expected[26:41, 26:41] = True
        np.testing.assert_array_equal(inset.inset_mask, expected)

    def test_after_positive_inset_a_nonzero_area_ring_can_still_self_touch(self):
        mask = blank()
        mask[4:13, 4:13] = True
        mask[12:21, 12:21] = True
        result = self.detector.detect(mask)
        inset = result.inset_component(1)
        self.assertEqual(inset.summary['status'], 'computed')
        self.assertEqual(inset.summary['polygon_status'], 'ambiguous')
        outer = inset.components[0]['outer']
        self.assertGreater(outer['area_px2'], 0)
        self.assertEqual(outer['status'], 'self_touch_or_crossing')
        self.assertEqual(inset.components[0]['polygon_reasons'], ['outer_ring_self_touch_or_crossing'])
        self.assertTrue(inset.inset_mask.any())
        tangent = result.inset_component(sqrt(2))
        self.assertEqual(tangent.summary['status'], 'split')
        self.assertFalse(tangent.inset_mask[12, 12])

    def test_thin_neck_keeps_all_surviving_parts(self):
        mask = blank()
        mask[8:37, 5:22] = True
        mask[13:32, 38:55] = True
        mask[22:23, 22:38] = True
        result = self.detector.detect(mask)
        inset = result.inset_component(3)
        self.assertEqual(inset.summary['status'], 'split')
        self.assertEqual(len(inset.components), 2)
        self.assertGreater(inset.components[0]['foreground_pixel_count'], inset.components[1]['foreground_pixel_count'])
        self.assertGreater(inset.components[1]['foreground_pixel_count'], 0)
        self.assertEqual(sum(p['foreground_pixel_count'] for p in inset.components), inset.inset_mask.sum())

    def test_hole_grows_and_nested_source_island_is_not_included(self):
        mask = blank()
        mask[5:47, 5:55] = True
        mask[17:34, 21:36] = False
        mask[21:30, 25:32] = True
        result = self.detector.detect(mask)
        inset = result.inset_component(2)
        self.assertEqual(inset.summary['status'], 'computed')
        self.assertEqual(inset.summary['hole_count'], 1)
        part = inset.components[0]
        self.assertEqual(part['polygon_status'], 'simple')
        np.testing.assert_array_equal(inset.inset_mask[17:34, 21:36], False)
        self.assertFalse(inset.inset_mask[16, 25])
        self.assertTrue(inset.inset_mask[14, 25])

        def signed(points):
            p = np.asarray(points)
            q = np.roll(p, -1, axis=0)
            return (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum()

        self.assertGreater(signed(part['outer']['points_xy']), 0)
        self.assertLess(signed(part['holes'][0]['points_xy']), 0)

    def test_inset_can_open_a_hole_or_make_rings_touch(self):
        mask = blank()
        mask[5:32, 5:35] = True
        mask[7:19, 11:28] = False
        result = self.detector.detect(mask)
        self.assertEqual(len(result.holes), 1)
        self.assertEqual(result.inset_component(1).summary['hole_count'], 0)

        mask[:] = False
        mask[3:17, 3:17] = True
        mask[6:14, 6:14] = False
        inset = self.detector.detect(mask).inset_component(1)
        part = inset.components[0]
        self.assertEqual(part['outer']['status'], 'simple')
        self.assertEqual(part['holes'][0]['status'], 'simple')
        self.assertEqual(part['polygon_status'], 'ambiguous')
        self.assertEqual(part['polygon_reasons'], ['rings_touch_or_cross'])

    def test_degenerate_output_is_not_empty_or_a_simple_polygon(self):
        mask = blank()
        mask[7:10, 9:30] = True
        result = self.detector.detect(mask)
        line = result.inset_component(1)
        self.assertEqual(line.summary['status'], 'computed')
        self.assertEqual(line.summary['polygon_status'], 'ambiguous')
        self.assertEqual(line.components[0]['outer']['status'], 'degenerate')
        self.assertEqual(line.inset_mask.sum(), 19)
        gone = result.inset_component(2)
        self.assertEqual(gone.summary['status'], 'collapsed')
        self.assertEqual(gone.summary['polygon_status'], 'unavailable')
        self.assertEqual(gone.components, [])
        self.assertFalse(gone.inset_mask.any())

    def test_partial_observation_and_frame_are_not_interpreted_as_material(self):
        mask = np.ones((22, 24), bool)
        observed = np.ones_like(mask)
        observed[:, :3] = False
        observed[8:12, 9:12] = False
        result = self.detector.detect(mask, observed_mask=observed)
        zero = result.inset_component(0)
        np.testing.assert_array_equal(zero.inset_mask, observed)
        inset = result.inset_component(1)
        self.assertTrue(inset.summary['source_contour_clipped'])
        self.assertTrue(inset.summary['source_touches_unobserved_or_frame'])
        self.assertFalse(inset.inset_mask[0].any())
        self.assertFalse(inset.inset_mask[:, :4].any())
        self.assertFalse(inset.inset_mask[~observed].any())
        self.assertFalse(inset.inset_mask[9, 12])
        self.assertTrue(inset.inset_mask[9, 13])

        interior = blank()
        interior[6:46, 6:51] = True
        observed = np.ones_like(interior)
        observed[22, 24] = False
        hidden_hole = self.detector.detect(interior, observed_mask=observed).inset_component(1)
        self.assertFalse(hidden_hole.summary['source_contour_clipped'])
        self.assertTrue(hidden_hole.summary['source_touches_unobserved_or_frame'])

    def test_select_by_id_not_index_and_validate_scalar(self):
        mask = blank()
        mask[6:17, 5:19] = True
        mask[20:47, 20:57] = True
        result = self.detector.detect(mask)
        result.contours[0]['id'], result.contours[1]['id'] = 8, 48
        self.assertEqual(result.inset_component(2).summary['contour_id'], 48)
        explicit = result.inset_component(np.float32(2), np.int64(8))
        self.assertEqual(explicit.summary['contour_id'], 8)
        self.assertFalse(explicit.inset_mask[20:].any())
        for bad in [-1, '1', None, complex(1), np.nan, np.inf, True, np.bool_(False), [2], 10**400]:
            with self.subTest(offset=bad), self.assertRaises(ValueError):
                result.inset_component(bad)
        for bad in [False, '1', 8., 0, 2, -1]:
            with self.subTest(contour=bad), self.assertRaises(ValueError):
                result.inset_component(2, bad)
        no_input = self.detector.detect(blank()).inset_component(2)
        self.assertEqual(no_input.summary['status'], 'no_visible_component')
        self.assertIsNone(no_input.summary['contour_id'])
        self.assertEqual(no_input.components, [])

    def test_distance_oracle_and_monotonicity(self):
        rng = np.random.default_rng(8)
        for _ in range(8):
            mask = rng.random((15, 17)) > .30
            mask[5:13, 5:13] = True
            result = self.detector.detect(mask)
            selected = result.component_mask()
            need = np.argwhere(~np.pad(selected, 1)) - 1
            lattice = np.indices(selected.shape).transpose(1, 2, 0)
            exact_distance = np.sqrt(np.min(np.sum((lattice[..., None, :] - need) ** 2, axis=-1), axis=-1))
            previous = selected
            for offset in [0., np.nextafter(1., 0.), 1., np.nextafter(sqrt(2), 0.), sqrt(2), 2.,
                           np.nextafter(sqrt(5), 0.), sqrt(5), 3.1, 100.]:
                with self.subTest(offset=offset):
                    inset = result.inset_component(offset)
                    expected = selected & (exact_distance > offset)
                    np.testing.assert_array_equal(inset.inset_mask, expected)
                    self.assertFalse(inset.inset_mask[~previous].any())
                    self.assertEqual(sum(p['foreground_pixel_count'] for p in inset.components), expected.sum())
                    previous = inset.inset_mask


if __name__ == '__main__':
    unittest.main()
