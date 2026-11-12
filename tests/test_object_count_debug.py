"""Group visualization checks; synthetic rendering is not field accuracy validation."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from depth_image_lens import OverheadObjectCounter
from depth_image_lens.debug import DebugResultMixin, depth_canvas
from depth_image_lens.object_count_debug import _group_color, _group_images, _label_groups
from .test_object_count import scene, paint, detect


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV extra not installed')
class ObjectCountDebugTests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.cv = cv2
        self.arrays = self.color, self.depth, _, _ = scene()
        paint(self.color, self.depth, np.s_[20:60, 10:85])
        paint(self.color, self.depth, np.s_[10:70, 35:60], (35, 35, 210), 960)
        self.result = detect(self.arrays)

    def texts(self, result, **options):
        with patch.object(self.cv, 'putText', wraps=self.cv.putText) as draw:
            result.render_debug(**options)
        return [call.args[1] for call in draw.call_args_list]

    def test_same_id_fragments_share_color_and_different_ids_do_not(self):
        result = self.result
        self.assertEqual(sorted(g['fragment_count'] for g in result.groups), [1, 2])
        overlay, classified = _group_images(result, self.color, 1.)
        for group in result.groups:
            pixels = result.group_mask(group['id'])
            expected = np.broadcast_to(_group_color(group['id']), (pixels.sum(), 3))
            np.testing.assert_array_equal(classified[pixels], expected)
            np.testing.assert_array_equal(overlay[pixels], expected)
        np.testing.assert_array_equal(overlay[~result.object_mask], self.color[~result.object_mask])
        self.assertTrue((classified[~result.object_mask] == 35).all())
        self.assertNotEqual(_group_color(1), _group_color(2))

    def test_alpha_zero_and_fraction_do_not_affect_solid_map(self):
        zero, classified = _group_images(self.result, self.color, 0.)
        np.testing.assert_array_equal(zero, self.color)
        half, other = _group_images(self.result, self.color, .5)
        np.testing.assert_array_equal(classified, other)
        mask = self.result.object_mask
        expected = np.rint(.5*self.color[mask]+.5*classified[mask]).astype(np.uint8)
        np.testing.assert_array_equal(half[mask], expected)

    def test_badges_label_both_fragments_and_stay_inside_image(self):
        image = np.zeros((160, 200, 3), np.uint8)
        labels = np.zeros(image.shape[:2], np.int32)
        labels[:50, :60] = 1
        labels[110:, 140:] = 1
        labels[65:95, 70:130] = 2
        with patch.object(self.cv, 'putText', wraps=self.cv.putText) as draw:
            self.assertEqual(_label_groups(image, labels, range(1, 3)), 2)
        self.assertEqual([call.args[1] for call in draw.call_args_list], ['G1', 'G1', 'G2'])
        for call in draw.call_args_list:
            x, y = call.args[2]
            self.assertTrue(0 <= x < 200 and 0 <= y < 160)

    def test_count_none_keeps_group_ids_and_marks_uncertainty(self):
        result = OverheadObjectCounter().detect_mask(
            self.color, self.result.object_mask, color_order='RGB')
        self.assertIsNone(result.count)
        texts = self.texts(result, color=self.color, panel=False, max_lines=1)
        self.assertTrue(any('count: None (unresolved)' in t and 'candidate_count: 2' in t for t in texts))
        self.assertIn('G1', texts)
        self.assertIn('G2', texts)
        self.assertTrue(any(t.startswith('G1*:') for t in texts))
        self.assertTrue(any(t.startswith('G2*:') for t in texts))

    def test_public_view_contains_exact_group_map_and_preserves_evidence(self):
        for diagnostics in (False, True):
            result = detect(self.arrays, diagnostics=diagnostics)
            for panel in (False, True):
                with self.subTest(diagnostics=diagnostics, panel=panel):
                    output = result.render_debug(self.color, panel=panel, group_labels=False)
                    evidence = DebugResultMixin.render_debug(result, self.color, panel=panel)
                    np.testing.assert_array_equal(output[-len(evidence):], evidence)
                    h, w = result.label_map.shape
                    scale = min(480/w, 480/h, (output.shape[1]//2-16)/w)
                    tw, th = max(2, round(w*scale)), max(2, round(h*scale))
                    left = output.shape[1]//2+(output.shape[1]//2-tw)//2
                    _, classified = _group_images(result, self.color, .35)
                    np.testing.assert_array_equal(output[72:72+th, left:left+tw],
                        self.cv.resize(classified, (tw, th), interpolation=self.cv.INTER_NEAREST))

    def test_group_labels_false_keeps_the_complete_legend(self):
        texts = self.texts(self.result, group_labels=False, panel=False, max_lines=1)
        self.assertNotIn('G1', texts)
        self.assertNotIn('G2', texts)
        self.assertTrue(any(t.startswith('G1:') for t in texts))
        self.assertTrue(any(t.startswith('G2:') for t in texts))

    def test_rgb_bgr_noncontiguous_and_readonly_inputs_are_not_mutated(self):
        result = self.result
        original = copy.deepcopy(result)
        color = self.color.copy()
        color.setflags(write=False)
        for value in vars(result).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        rgb = result.render_debug(color)
        bgr = result.render_debug(color[..., ::-1], color_order='BGR')
        np.testing.assert_array_equal(bgr, rgb[..., ::-1])
        self.assertTrue(bgr.flags.c_contiguous)
        np.testing.assert_array_equal(color, self.color)
        self.assertEqual(result.to_dict(), original.to_dict())
        for key, value in vars(result).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(original, key))
        flipped = replace(result, **{key: value[:, ::-1] for key, value in vars(result).items()
                                    if isinstance(value, np.ndarray)})
        out = flipped.render_debug(color[:, ::-1, ::-1], color_order='BGR')
        self.assertEqual(out.dtype, np.uint8)

    def test_depth_and_no_source_views_and_missing_depth_colors(self):
        depth = self.depth.astype(float)
        depth[:4, :4] = np.nan
        before = depth.copy()
        rgb = self.result.render_debug(depth=depth)
        np.testing.assert_array_equal(self.result.render_debug(depth=depth, color_order='BGR'), rgb[..., ::-1])
        evidence = DebugResultMixin.render_debug(self.result, depth=depth)
        # The shared evidence titles identify depth vs RGB sources differently.
        np.testing.assert_array_equal(rgb[:-len(evidence)],
                                      self.result.render_debug(depth_canvas(depth))[:-len(evidence)])
        np.testing.assert_array_equal(depth, before)
        self.assertEqual(self.result.render_debug().dtype, np.uint8)
        np.testing.assert_array_equal(self.result.render_debug(color_order='BGR'),
                                      self.result.render_debug()[..., ::-1])

    def test_empty_and_all_excluded_and_minimum_grid(self):
        empty = detect(scene((2, 2)))
        self.assertEqual(empty.count, 0)
        self.assertTrue(any('count: 0 | candidate_count: 0' in t for t in self.texts(empty, panel=False)))
        excluded = detect(scene(), exclusion_mask=np.ones((80, 100), bool))
        self.assertIsNone(excluded.count)
        self.assertEqual(excluded.candidate_count, 0)
        texts = self.texts(excluded, panel=False)
        self.assertTrue(any('count: None (unresolved)' in t for t in texts))
        self.assertTrue(any('No retained groups' in t for t in texts))

    def test_all_128_groups_have_distinct_stable_colors_and_complete_legend(self):
        labels = np.repeat(np.repeat(np.arange(1, 129).reshape(8, 16), 16, axis=0), 16, axis=1).astype(np.int32)
        base = detect(scene(labels.shape))
        groups = [dict(id=i, area_px=256, fragment_count=1, reasons=[]) for i in range(1, 129)]
        result = replace(base, count=None, candidate_count=128, label_map=labels, groups=groups,
                         object_mask=labels > 0, candidate_mask=labels > 0, state='ambiguous')
        colors = [_group_color(i) for i in range(1, 129)]
        self.assertEqual(len(set(colors)), 128)
        texts = self.texts(result, panel=False, max_lines=1)
        for i in range(1, 129):
            self.assertTrue(any(t.startswith(f'G{i}:') for t in texts), i)
        image = result.render_debug(panel=False)
        np.testing.assert_array_equal(result.render_debug(panel=False), image)
        # The first group's display color must not depend on total group count.
        _, classified = _group_images(result, np.zeros((*labels.shape, 3), np.uint8), 1.)
        np.testing.assert_array_equal(classified[0, 0], _group_color(1))
        self.assertLess(image.shape[0], 2500)

    def test_tiny_crowded_fragments_omit_badges_without_losing_legend(self):
        labels = np.tile(np.arange(1, 21, dtype=np.int32), (2, 1))
        image = np.zeros((*labels.shape, 3), np.uint8)
        self.assertEqual(_label_groups(image, labels, range(1, 21)), 0)
        base = detect(scene(labels.shape))
        result = replace(base, candidate_count=20, label_map=labels,
                         groups=[dict(id=i, area_px=2, fragment_count=1, reasons=[]) for i in range(1, 21)])
        texts = self.texts(result, panel=False)
        self.assertTrue(any(t.startswith('G20:') for t in texts))
        self.assertTrue(any('tiny/crowded fragments' in t for t in texts))

    def test_validation_preserves_existing_contracts_and_checks_group_labels(self):
        for options in ({'color_order': 'auto'}, {'panel': 1}, {'max_lines': 0},
                        {'mask_alpha': 2}, {'mask_alpha': np.nan}, {'depth_scale_m': 0},
                        {'group_labels': 1}, {'group_labels': np.bool_(True)},
                        {'color': np.zeros((3, 4, 3), np.uint8)},
                        {'color': self.color.astype(float)}, {'depth': self.depth[:-1]}):
            with self.subTest(options=options.keys()), self.assertRaises(ValueError):
                self.result.render_debug(**options)
        for labels in (self.result.label_map[:-1], self.result.label_map.astype(float),
                       np.full_like(self.result.label_map, -1), np.full_like(self.result.label_map, 3)):
            with self.assertRaisesRegex(ValueError, 'label_map'):
                replace(self.result, label_map=labels).render_debug()

    def test_save_rgb_and_bgr_roundtrip_and_overwrite_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'groups.png'
            expected = self.result.render_debug(self.color, panel=False, group_labels=False)
            self.assertEqual(self.result.save_debug(dest, self.color, panel=False, group_labels=False), dest)
            np.testing.assert_array_equal(self.cv.imread(str(dest))[..., ::-1], expected)
            with self.assertRaises(FileExistsError):
                self.result.save_debug(dest, self.color)
            self.result.save_debug(dest, self.color[..., ::-1], color_order='BGR',
                                   panel=False, group_labels=False, overwrite=True)
            np.testing.assert_array_equal(self.cv.imread(str(dest))[..., ::-1], expected)

    def test_example_runs_and_saves_all_four_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run([sys.executable, 'examples/example_object_count_debug.py',
                                      '--output-dir', tmp], check=True, capture_output=True, text=True)
            summaries = json.loads(process.stdout)
            self.assertEqual(summaries['groups']['count'], 2)
            self.assertIsNone(summaries['mask-only']['count'])
            self.assertIsNone(summaries['uncertain']['count'])
            self.assertEqual(summaries['empty']['count'], 0)
            for name in summaries:
                image = self.cv.imread(str(Path(tmp)/f'object-count-{name}.png'))
                self.assertIsNotNone(image)
                self.assertEqual(image.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
