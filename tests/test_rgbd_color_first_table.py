"""Synthetic opt-in table-anchor contracts, not sensor validation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import detect


ANCHORS = [[14, 14], [28, 17], [85, 17], [99, 14]]


def scene(offset=6, height=2):
    empty_color = np.full((96, 112, 3), 140, np.uint8)
    empty_depth = np.full((96, 112), 1000, np.uint16)
    color = np.full_like(empty_color, 100)  # whole table a color candidate
    depth = np.full_like(empty_depth, 1000-offset)
    obj = np.zeros(depth.shape, bool)
    obj[34:66, 36:76] = True
    color[obj] = (205, 70, 125)
    depth[obj] -= height
    return (color, depth, empty_color, empty_depth), obj


def run(args, anchors=ANCHORS, *, debug=True, mode='color_first', **options):
    return detect(*args, RGBDBackgroundSegmenter(
        fusion_mode=mode, first_table_xy=anchors, **options), diagnostics=debug)


class TableAnchorTests(unittest.TestCase):
    def test_table_shift_with_no_legacy_ring(self):
        for bias in (6, -6):
            with self.subTest(bias=bias):
                args, obj = scene(bias, height=1)
                legacy = run(args, None)
                r = run(args)
                self.assertFalse(legacy.object_mask.any())
                np.testing.assert_array_equal(r.object_mask, obj)
                np.testing.assert_array_equal(r.confident_background_mask, ~obj)
                self.assertTrue(r.inferred_foreground_mask[obj].all())
                self.assertTrue(r.diagnostics['first_candidate_mask'].all())
                self.assertFalse(r.unknown_mask[~obj].any())
                table = r.stages['fusion']['table_reference']
                self.assertEqual(table['status'], 'accepted')
                self.assertAlmostEqual(table['offset_m'], bias*.001)
                ref = r.stages['fusion']['regions'][0]
                self.assertEqual(ref['table_reference_source'], 'anchor_local_ring')
                self.assertAlmostEqual(ref['median_delta_m'], .001)
                self.assertEqual(r.stages['fusion']['decision_reasons']['8'], 'anchor_supported_table_background')
                self.assertEqual(bool(r.diagnostics['first_table_raw_depth_foreground_mask'][0,0]), bias>5)
                json.dumps(r.to_dict(), allow_nan=False)

    def test_raw_anchor_depth_missing_or_outlier_is_not_a_vote(self):
        args, obj = scene(3)
        c,z,bc,bz = args
        z[14,14] = 0
        z[17,28] = 910
        bz[17,85] = 0
        r = run(args)
        ref = r.stages['fusion']['table_reference']
        self.assertEqual(ref['votes'], 4)
        self.assertAlmostEqual(ref['offset_m'], .003)
        self.assertTrue(r.unknown_mask[14,14])
        self.assertTrue(r.unknown_mask[17,85])
        self.assertFalse(r.confident_background_mask[17,28])
        self.assertFalse(r.diagnostics['first_table_reference_mask'][17,28])
        np.testing.assert_array_equal(r.object_mask, obj)

    def test_mean_vs_median_for_signed_outlier(self):
        args, obj = scene(3)
        args[1][21:26,20:30] = 967  # minority 30mm residual outliers in the table core
        median = run(args)
        mean = run(args, first_table_statistic='mean')
        md = median.stages['fusion']['table_reference']
        mn = mean.stages['fusion']['table_reference']
        self.assertEqual(mn['status'], 'accepted')
        self.assertAlmostEqual(md['offset_m'], .003)
        self.assertGreater(mn['offset_m'], md['offset_m']+.0001)
        self.assertEqual(mn['offset_m'], mn['sample_mean_m'])
        np.testing.assert_array_equal(mean.object_mask, obj)
        np.testing.assert_array_equal(median.object_mask, obj)

    def test_anchors_outside_or_color_invalid(self):
        args, _ = scene()
        r = run(args, [[300,500], [100,30]])
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'insufficient_anchors')
        self.assertFalse(r.confident_background_mask.any())
        self.assertFalse(r.diagnostics['first_table_forced_background_mask'].any())
        r = run(args, [[14,14], [15,14]])
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'anchors_too_close')

    def test_one_cloth_vote_preserves_table_consensus(self):
        args, obj = scene(3)
        r = run(args, ANCHORS[:3]+[[45,40]])
        self.assertEqual(r.stages['fusion']['table_reference']['votes'], 3)
        np.testing.assert_array_equal(r.object_mask, obj)

    def test_occluded_side_and_large_cloth_are_not_calibration(self):
        args, _ = scene(3)
        c,z,_,_ = args
        c[:30,75:] = (205,70,125)
        z[:30,75:] = 964
        r = run(args)
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'no_shared_color_region')
        self.assertFalse(r.confident_background_mask.any())
        c[:] = (205,70,125)
        z[:] = 964
        r = run(args)
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'table_chroma_changed')
        self.assertFalse(r.confident_background_mask.any())
        self.assertIsNone(r.stages['fusion']['table_reference']['offset_m'])

    def test_dissimilar_thin_cloth_over_all_anchors_does_not_become_table(self):
        args, _ = scene(0)
        c,z,bc,bz = args
        c[:] = bc
        z[:] = bz
        c[5:60,6:105] = (205,70,125)
        z[5:60,6:105] = 998
        base,r = run(args,None),run(args)
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'table_chroma_changed')
        self.assertFalse(r.confident_background_mask.any())
        self.assertTrue(base.object_mask[10,20])
        np.testing.assert_array_equal(r.object_mask, base.object_mask)
        z[:],c[:] = 970,90  # near-grey thick cloth, chroma guard does not apply
        r = run(args)
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'offset_out_of_range')

    def test_insufficient_valid_depth_uses_legacy(self):
        args, obj = scene()
        args[1][~obj] = 0
        r, base = run(args), run(args, None)
        self.assertEqual(r.stages['fusion']['table_reference']['status'], 'insufficient_valid_depth')
        np.testing.assert_array_equal(r.object_mask, base.object_mask)
        self.assertFalse(r.confident_background_mask.any())

    def test_spatial_mismatch_does_not_apply_scalar_offset(self):
        args, obj = scene(1)
        args[1][:,56:][~obj[:,56:]] = 991
        r, base = run(args), run(args, None)
        self.assertIn(r.stages['fusion']['table_reference']['status'],
                      ('inconsistent_table_depth','offset_out_of_range'))
        self.assertFalse(r.confident_background_mask.any())
        np.testing.assert_array_equal(r.object_mask, base.object_mask)

    def test_shadow_not_foreground_and_wider_table_reference(self):
        args, obj = scene(3)
        shadow = np.zeros(obj.shape,bool)
        shadow[17:83,19:93] = True
        shadow &= ~obj
        args[0][shadow] = (40,40,40)
        # Anchors remain on measured table on both sides.
        r = run(args, [[9,15],[15,15],[98,15],[103,15]])
        np.testing.assert_array_equal(r.object_mask, obj)
        records = r.stages['fusion']['regions']
        target = next(x for x in records if x['accepted'])
        self.assertEqual(target['table_reference_source'], 'anchor_expanded_ring')
        self.assertEqual(target['table_reference_radius_px'], 48)

    def test_similarly_colored_elevated_cloth_cannot_contaminate_reference(self):
        shape = (150, 240)
        bc = np.full((*shape,3),130,np.uint8)
        bz = np.full(shape,1000.,float)
        c,z = bc.copy(),bz.copy()
        a,b = [np.zeros(shape,bool) for _ in range(2)]
        a[60:96,95:133] = True
        b[44:112,81:149] = True
        b &= ~a
        b[44:63,95:134] = False
        c[a], z[a] = (205,70,125), 997.5
        c[b], z[b] = 145, 998.
        r = run((c,z,bc,bz), [[14,14],[28,17],[206,17],[225,14]])
        np.testing.assert_array_equal(r.object_mask,a|b)
        self.assertFalse(r.diagnostics['first_table_reference_mask'][b].any())
        self.assertFalse(r.confident_background_mask[b].any())
        self.assertTrue(r.diagnostics['first_table_proposal_mask'][b].all())

    def test_unshifted_color_table_can_vote_without_color_change(self):
        args, obj = scene(3)
        args[0][~obj] = args[2][~obj]
        r = run(args)
        self.assertFalse(r.diagnostics['first_candidate_mask'][~obj].any())
        self.assertEqual(r.stages['fusion']['table_reference']['votes'], 4)
        np.testing.assert_array_equal(r.object_mask, obj)

    def test_invalid_inputs_and_non_color_first_isolation(self):
        bad = (
            dict(first_table_xy=[[2,2],[2,2]]),
            dict(first_table_xy=[[2.5,2],[8,9]]),
            dict(first_table_xy=[[True,2],[8,9]]),
            dict(first_table_statistic='trim'),
            dict(first_table_max_offset_m=0),
            dict(first_table_max_deviation_m=float('inf')),
            dict(first_table_min_anchor_span_px=True),
        )
        for kw in bad:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                RGBDBackgroundSegmenter(**kw)
        args, _ = scene(3)
        for mode in ('union','depth_required','color_supported','color_model'):
            a,b = run(args,None,mode=mode),run(args,mode=mode)
            self.assertEqual(a.stages,b.stages)
            np.testing.assert_equal(a.diagnostics,b.diagnostics)
            np.testing.assert_array_equal(a.object_mask,b.object_mask)

    def test_render_export_and_tuner_crop_local_coordinates(self):
        from depth_image_lens.tuning_backend import (
            default_config, inspect_pixel, load_config, prepare_inputs, run_pipeline, save_config)
        args, obj = scene()
        r = run(args)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            r.save_debug(path/'view.png', color=args[0], color_order='RGB')
            r.save_debug_bundle(path/'bundle', color=args[0], color_order='RGB')
            self.assertGreater((path/'view.png').stat().st_size, 0)
            self.assertTrue((path/'bundle'/'index.json').is_file())
            conf = default_config()
            conf['diagnostics'] = True
            conf['segmenter'].update(fusion_mode='color_first', first_table_xy=ANCHORS)
            save_config(path/'fixture.json', conf)
            conf = load_config(path/'fixture.json')
            self.assertEqual(conf['segmenter']['first_table_xy'], ANCHORS)
            padded = {}
            for role,array in zip(('color','depth','background_color','background_depth'), args):
                dest = np.empty((400,640)+array.shape[2:], dtype=array.dtype)
                dest[...] = array[0,0]
                dest[23:119,41:153] = array
                padded[role] = dest
            res = run_pipeline(prepare_inputs(padded,(41,23,112,96)),conf)
            np.testing.assert_array_equal(res.object_mask,obj)
            key = 'segmenter/first_table_forced_background_mask'
            self.assertTrue(inspect_pixel(res,key,14,14)['value'])
            self.assertEqual(inspect_pixel(res,key,14,14)['normalized_xy'],[55,37])
            self.assertEqual(res.summary['stages']['segmenter']['stages']['fusion']
                             ['table_reference']['anchor_xy'], ANCHORS)

    def test_debug_does_not_change_decisions(self):
        args, _ = scene(3)
        a,b = run(args),run(args,debug=False)
        np.testing.assert_array_equal(a.object_mask,b.object_mask)
        np.testing.assert_array_equal(a.confident_background_mask,b.confident_background_mask)
        self.assertEqual(a.stages,b.stages)


if __name__ == '__main__':
    unittest.main()
