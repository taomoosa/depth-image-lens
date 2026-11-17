"""Full debug export coverage and persistence contracts."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from depth_image_lens.debug import DebugResultMixin


class ExampleResult(DebugResultMixin):
    def __init__(self, diagnostics=None, shape=(20, 32)):
        self.object_mask = np.zeros(shape, bool)
        self.object_mask[4:16, 5:26] = True
        self.diagnostics = diagnostics or {}
        self.parameters = {'depth_threshold_m': .003}


class DebugBundleTests(unittest.TestCase):
    def export(self, result, root, **kwargs):
        path = result.save_debug_bundle(Path(root) / 'bundle', **kwargs)
        self.assertIsInstance(path, Path)
        self.assertTrue(path.is_file())
        return path, json.loads(path.read_text())

    def test_all_general_arrays_without_truncation(self):
        diagnostics = {f'layer_{i}_mask': np.full((20, 32), i % 2, bool)
                       for i in range(31)}
        with tempfile.TemporaryDirectory() as root:
            path, data = self.export(ExampleResult(diagnostics), root, tile_width=160)
            entries = [e for e in data['entries'] if e['status'] == 'rendered']
            for key in diagnostics:
                self.assertTrue(any(e['key'].endswith(key) for e in entries), key)
            pages = {e['page'] for e in entries}
            self.assertGreaterEqual(len(pages), 2)
            self.assertTrue(all((path.parent / p).is_file() for p in pages))
            self.assertTrue(all(0 <= e['row'] < 4 and 0 <= e['column'] < 6 for e in entries))

    def test_bool_labels_numeric_nan_and_unsupported_are_accounted(self):
        numeric = np.arange(640, dtype=float).reshape(20, 32)
        numeric[0, 0] = np.nan
        diagnostics = {'region_label_map': np.ones((20, 32), np.int32),
                       'depth_difference_m': numeric,
                       'unused_vector': np.arange(7),
                       'diagnostic_metadata': {'maps': {'depth_difference_m': {'unit': 'm'}}}}
        original = numeric.copy()
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(ExampleResult(diagnostics), root, tile_width=160)
            for key in ('region_label_map', 'depth_difference_m'):
                self.assertTrue(any(e['key'].endswith(key) and e['status'] == 'rendered'
                                    for e in data['entries']))
            self.assertTrue(any(e['key'].endswith('unused_vector') and e['status'].startswith('unsupported')
                                for e in data['entries']))
        np.testing.assert_array_equal(original, numeric)

    def test_default_native_width_and_custom_pagination(self):
        import cv2
        with tempfile.TemporaryDirectory() as root:
            path, data = self.export(ExampleResult(shape=(400, 640)), root)
            entry = next(e for e in data['entries'] if e['status'] == 'rendered')
            image = cv2.imread(str(path.parent / entry['page']))
            self.assertGreaterEqual(image.shape[1], 640)
            self.assertGreaterEqual(image.shape[0], 400)
        diagnostics = {f'layer_{i}': np.zeros((20, 32), bool) for i in range(7)}
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(ExampleResult(diagnostics), root, columns=2, max_rows=2, tile_width=160)
            entries = [e for e in data['entries'] if e['status'] == 'rendered']
            self.assertGreaterEqual(len({e['page'] for e in entries}), 2)
            self.assertTrue(all(e['row'] < 2 and e['column'] < 2 for e in entries))

    def test_stack_maps_cover_every_model(self):
        diagnostics = {'model_diagnostic_ids': np.array([11, 22, 33, 44]),
            'model_all_distance': np.ones((4, 20, 32), np.float32),
            'model_all_candidate_mask': np.ones((4, 20, 32), bool)}
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(ExampleResult(diagnostics), root, tile_width=160)
            entries = [e for e in data['entries'] if e['status'] == 'rendered' and 'model_all_' in e['key']]
            self.assertEqual(len(entries), 8)
            self.assertEqual({e['model_id'] for e in entries}, {11, 22, 33, 44})
            for page in {e['page'] for e in entries}:
                self.assertLessEqual(len({e['model_id'] for e in entries if e['page'] == page}), 3)

    def test_model_filter_and_unavailable_maps_are_explicit(self):
        diagnostics = {'model_diagnostic_ids': np.array([11, 22]),
            'model_all_distance': np.ones((2, 20, 32), np.float32),
            'diagnostic_metadata': {'maps': {'missing_mask': {'unit': 'boolean'}}}}
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(ExampleResult(diagnostics), root,
                                  tile_width=160, models=[22])
            models = [e for e in data['entries'] if 'model_id' in e]
            self.assertEqual({e['model_id'] for e in models if e['status'] == 'rendered'}, {22})
            self.assertTrue(any(e['model_id'] == 11 and 'omitted' in e['status'] for e in models))
            self.assertTrue(any(e['key'].endswith('missing_mask') and e['status'] == 'not_acquired'
                                for e in data['entries']))
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(ExampleResult(), root, tile_width=160)
            self.assertTrue(any(e['key'] == 'diagnostics' and e['status'] == 'not_acquired'
                                for e in data['entries']))

    def test_rejection_bit_conditions_are_not_collapsed(self):
        from depth_image_lens._rgbd_color_model import MODEL_REJECTION_FLAGS
        bits = list(MODEL_REJECTION_FLAGS)
        flags = np.full((1, 20, 32), bits[0] | bits[1], np.uint16)
        diagnostics = {'model_diagnostic_ids': np.array([5]),
                       'model_all_rejection_flags': flags}
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(ExampleResult(diagnostics), root, tile_width=160)
            models = [e for e in data['entries'] if e.get('model_id') == 5]
            for reason in MODEL_REJECTION_FLAGS.values():
                self.assertTrue(any(e['key'].endswith('/' + reason) and e['status'] == 'rendered'
                                    for e in models), reason)
        np.testing.assert_array_equal(flags, bits[0] | bits[1])

    def test_existing_destination_and_symlink_rejected(self):
        result = ExampleResult()
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / 'bundle'
            dest.mkdir()
            marker = dest / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaises((FileExistsError, ValueError)):
                result.save_debug_bundle(dest)
            self.assertEqual(marker.read_text(), 'keep')
            link = Path(root) / 'link'
            link.symlink_to(dest, target_is_directory=True)
            with self.assertRaises((FileExistsError, ValueError)):
                result.save_debug_bundle(link)

    def test_invalid_options_and_mismatched_source(self):
        for options in ({'columns': 0}, {'max_rows': 0}, {'tile_width': 0},
                        {'columns': True}, {'models': [999]}, {'color_order': 'wrong'}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    ExampleResult().save_debug_bundle(Path(root) / 'bundle', **options)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                ExampleResult().save_debug_bundle(Path(root) / 'bundle', np.zeros((21, 32, 3), np.uint8))

    def test_color_first_maps_and_region_statistics_survive_export(self):
        from depth_image_lens import RGBDBackgroundSegmenter
        bg = np.full((64, 96, 3), 180, np.uint8)
        color = bg.copy(); color[16:48, 24:72] = 80
        bg_depth = np.full((64, 96), 1000, np.uint16)
        depth = bg_depth.copy(); depth[16:48, 24:72] = 998
        result = RGBDBackgroundSegmenter(fusion_mode='color_first').detect(
            color, depth, background_color=bg, background_depth=bg_depth,
            color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
            diagnostics=True)
        before = result.object_mask.copy()
        with tempfile.TemporaryDirectory() as root:
            _, data = self.export(result, root, color=color, tile_width=160)
            keys = {e['key'] for e in data['entries'] if e['status'] == 'rendered'}
            for name in ('first_candidate_mask', 'first_color_group_map',
                         'first_region_label_map', 'first_interior_mask',
                         'first_surround_mask', 'first_mean_delta_m',
                         'first_median_delta_m', 'first_required_delta_m'):
                self.assertIn('diagnostics/' + name, keys)
            self.assertEqual(data['stages']['fusion']['regions'],
                             result.stages['fusion']['regions'])
        np.testing.assert_array_equal(before, result.object_mask)

    def test_original_render_debug_unchanged(self):
        result = ExampleResult()
        before = result.render_debug(panel=False)
        with tempfile.TemporaryDirectory() as root:
            self.export(result, root, tile_width=160)
        np.testing.assert_array_equal(before, result.render_debug(panel=False))


if __name__ == '__main__':
    unittest.main()
