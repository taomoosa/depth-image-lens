"""Color-first weak-depth contracts; synthetic cases are not sensor accuracy."""
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import scene, detect

class ColorFirstTests(unittest.TestCase):
    def run_first(self, c,z,bc,bz,**kw):
        return detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_first',**kw),diagnostics=True)

    def object(self, mm=2):
        c,z,bc,bz=scene(); c[8:32,10:34]=80; z[8:32,10:34]=1000-mm
        return c,z,bc,bz

    def test_thin_without_strong_pixel_and_full_boundary(self):
        r=self.run_first(*self.object())
        self.assertFalse(r.depth_foreground_mask.any())
        self.assertEqual(r.object_mask.sum(),576)
        self.assertTrue(r.inferred_foreground_mask[r.object_mask].all())
        self.assertTrue(r.unknown_mask[r.object_mask].all())
        self.assertFalse(r.diagnostics['first_interior_mask'][8,15])

    def test_integer_mm_and_global_reference_bias(self):
        for mm in (1,2,3):
            c,z,bc,bz=self.object(mm);z-=2
            r=self.run_first(c,z,bc,bz)
            self.assertEqual(r.object_mask.sum(),576)

    def test_shadow_and_global_bias_rejected(self):
        for bias in (0,2,4):
            c,z,bc,bz=self.object(0);z-=bias
            r=self.run_first(c,z,bc,bz)
            self.assertFalse(r.object_mask.any())
            self.assertTrue(r.diagnostics['first_failed_median_delta_mask'].any())

    def test_no_change_empty(self):
        r=self.run_first(*scene());self.assertFalse(r.object_mask.any())

    def test_signed_negative_outliers_not_positive_only_average(self):
        c,z,bc,bz=self.object(0)
        z[8:32:2,10:34]=997;z[9:32:2,10:34]=1003
        r=self.run_first(c,z,bc,bz);self.assertFalse(r.object_mask.any())

    def test_sparse_positive_spikes_fail_median(self):
        c,z,bc,bz=self.object(0);z[8:32:4,10:34]=992
        r=self.run_first(c,z,bc,bz);self.assertFalse(r.object_mask.any())

    def test_missing_depth_never_filled(self):
        c,z,bc,bz=self.object();z[18:20,18:20]=0
        r=self.run_first(c,z,bc,bz)
        self.assertTrue(r.object_mask[10,12]);self.assertFalse(r.object_mask[18:20,18:20].any())

    def test_insufficient_interior_unknown(self):
        c,z,bc,bz=scene();c[8:32,20:23]=80;z[8:32,20:23]=998
        r=self.run_first(c,z,bc,bz)
        self.assertFalse(r.object_mask.any());self.assertTrue(r.diagnostics['first_insufficient_mask'].any())

    def test_no_surround_unknown(self):
        c,z,bc,bz=scene();c[:]=80;z[:]=998
        r=self.run_first(c,z,bc,bz);self.assertFalse(r.object_mask.any())
        self.assertTrue(r.diagnostics['first_failed_surround_blocks_mask'].all())

    def sparse_surround(self, kind='strip', mm=2):
        args = self.object(mm)
        obj = np.zeros(args[1].shape, bool)
        obj[8:32,10:34] = True
        ring = np.zeros_like(obj)
        if kind == 'strip':
            ring[3:5,8:40] = True  # 64 pixels, no half-filled 8x8 tiles
        elif kind == 'one_block':
            ring[:4,16:24] = True  # one half-filled 8x8 tile
        elif kind == 'tiny':
            ring[4,15] = True
        return args, obj, ring

    def run_sparse(self, args, obj, ring, minimum, **kwargs):
        return detect(*args, RGBDBackgroundSegmenter(
            fusion_mode='color_first', first_min_surround_blocks=minimum, **kwargs),
            work_roi=obj | ring, diagnostics=True)

    def test_surround_override_zero_recovers_no_qualifying_blocks(self):
        args, obj, ring = self.sparse_surround()
        legacy = self.run_sparse(args, obj, ring, None)
        for minimum, accepted in ((1, False), (0, True)):
            with self.subTest(minimum=minimum):
                r = self.run_sparse(args, obj, ring, minimum)
                stats = r.stages['fusion']['regions'][0]
                self.assertEqual(stats['surround_blocks'], 0)
                self.assertEqual(stats['surround_min_blocks'], minimum)
                self.assertEqual(stats['surround_valid_pixels'], 64)
                self.assertEqual(stats['surround_reference_source'], 'ring_pixels' if accepted else None)
                self.assertEqual(bool(r.object_mask.any()), accepted)
                self.assertEqual(bool(r.diagnostics['first_failed_surround_blocks_mask'].any()), not accepted)
                for key in ('first_candidate_mask','first_region_label_map','first_interior_mask','first_surround_mask'):
                    np.testing.assert_array_equal(r.diagnostics[key], legacy.diagnostics[key])
        np.testing.assert_array_equal(r.object_mask, obj)
        np.testing.assert_array_equal(r.inferred_foreground_mask, obj)
        self.assertIsNone(legacy.diagnostics['first_statistics'][0]['median_delta_m'])
        self.assertEqual(stats['surround_min_reference_pixels'], 20)

    def test_surround_override_uses_measured_one_block_spread(self):
        args, obj, ring = self.sparse_surround('one_block')
        for minimum, accepted in ((None, False), (2, False), (1, True), (0, True)):
            with self.subTest(minimum=minimum):
                r = self.run_sparse(args, obj, ring, minimum)
                self.assertEqual(r.diagnostics['first_statistics'][0]['surround_blocks'], 1)
                self.assertEqual(bool(r.object_mask.any()), accepted)
        # Block mean is still zero; measured pixel spread is 1 mm.
        args[1][:2,16:24] = 999
        args[1][2:4,16:24] = 1001
        r = self.run_sparse(args, obj, ring, 1)
        stats = r.diagnostics['first_statistics'][0]
        self.assertEqual(stats['surround_reference_source'], 'ring_pixels')
        self.assertAlmostEqual(stats['surround_noise_m'], .001)
        self.assertAlmostEqual(stats['required_delta_m'], .00325)
        self.assertFalse(r.object_mask.any())

    def test_surround_override_three_blocks_uses_block_means(self):
        args, obj, ring = self.sparse_surround('empty')
        ring[:4,8:32] = True
        legacy = self.run_sparse(args, obj, ring, None)
        relaxed = self.run_sparse(args, obj, ring, 3)
        self.assertFalse(legacy.object_mask.any())
        np.testing.assert_array_equal(relaxed.object_mask, obj)
        stats = relaxed.diagnostics['first_statistics'][0]
        self.assertEqual(stats['surround_blocks'], 3)
        self.assertEqual(stats['surround_reference_source'], 'block_means')
        self.assertIsNone(stats['surround_min_reference_pixels'])

    def test_surround_override_empty_invalid_and_tiny_references(self):
        for kind, invalidate, count in (('empty', False, 0), ('strip', True, 0), ('tiny', False, 1)):
            with self.subTest(kind=kind, invalidate=invalidate):
                args, obj, ring = self.sparse_surround(kind)
                if invalidate:
                    args[1][ring] = 0
                r = self.run_sparse(args, obj, ring, 0)
                stats = r.diagnostics['first_statistics'][0]
                self.assertEqual(stats['surround_valid_pixels'], count)
                self.assertIsNone(stats['surround_reference_source'])
                self.assertIsNone(stats['required_delta_m'])
                self.assertFalse(r.object_mask.any())
                self.assertTrue(r.unknown_mask[obj].all())
                self.assertTrue(r.diagnostics['first_failed_surround_reference_mask'][obj].all())
                self.assertFalse(r.diagnostics['first_failed_surround_blocks_mask'].any())
                self.assertEqual('surround_coverage' in stats['failures'], count == 0)

    def test_surround_override_fallback_floor_is_inclusive(self):
        args, obj, ring = self.sparse_surround('empty')
        for count in (19, 20):
            with self.subTest(count=count):
                ring[4,10:10+count] = True
                r = self.run_sparse(args, obj, ring, 0)
                stats = r.diagnostics['first_statistics'][0]
                self.assertEqual(stats['surround_valid_pixels'], count)
                self.assertEqual(stats['surround_blocks'], 0)
                self.assertEqual(stats['surround_valid_fraction'], 1.0)
                self.assertEqual(bool(r.object_mask.any()), count == 20)

    def test_surround_override_keeps_coverage_and_interior_blocks(self):
        args, obj, ring = self.sparse_surround()
        args[1][3,8:40] = 0  # coverage fails even with enough fallback pixels
        r = self.run_sparse(args, obj, ring, 0)
        self.assertEqual(r.diagnostics['first_statistics'][0]['failures'], ['surround_coverage'])
        self.assertFalse(r.object_mask.any())
        args, obj, ring = self.sparse_surround()
        r = self.run_sparse(args, obj, ring, 0, first_min_blocks=8)
        self.assertEqual(r.diagnostics['first_statistics'][0]['failures'], ['region_blocks'])
        self.assertFalse(r.object_mask.any())
        args[1][8:22,10:34] = 0
        r = self.run_sparse(args, obj, ring, 0)
        self.assertTrue(r.diagnostics['first_failed_region_coverage_mask'][obj].all())
        self.assertFalse(r.object_mask.any())

    def test_surround_override_keeps_relative_depth_gates(self):
        for case in ('shadow', 'mean_delta', 'median_delta', 'block_fraction'):
            with self.subTest(case=case):
                args, obj, ring = self.sparse_surround(mm=0 if case in ('shadow', 'median_delta') else 2)
                z = args[1]
                if case == 'mean_delta':
                    z[8:32:4,10:34] = 1010  # positive median, signed mean fails
                elif case == 'median_delta':
                    z[8:32:4,10:34] = 992  # positive mean, zero median
                elif case == 'block_fraction':
                    z[obj] = 997
                    z[8:32,24:32] = 1000  # both pixel deltas pass; too few blocks agree
                else:
                    z -= 3  # coherent global bias still cancels for a color-only shadow
                r = self.run_sparse(args, obj, ring, 0)
                stats = r.diagnostics['first_statistics'][0]
                self.assertEqual(stats['surround_reference_source'], 'ring_pixels')
                self.assertFalse(r.object_mask.any())
                self.assertIn('median_delta' if case == 'shadow' else case, stats['failures'])
                if case in ('median_delta', 'block_fraction'):
                    self.assertEqual(stats['failures'], [case])

    def test_surround_override_qualified_ring_keeps_existing_statistics(self):
        args = self.object()
        args[1][:8,::2] = 999
        legacy = self.run_first(*args)
        explicit_none = self.run_first(*args, first_min_surround_blocks=None)
        override = self.run_first(*args, first_min_surround_blocks=0)
        orig = legacy.diagnostics['first_statistics'][0]
        self.assertEqual(legacy.stages, explicit_none.stages)
        for result in (explicit_none, override):
            np.testing.assert_array_equal(result.object_mask, legacy.object_mask)
            new = result.diagnostics['first_statistics'][0]
            self.assertEqual({k:new[k] for k in orig}, orig)
        self.assertEqual(new['surround_reference_source'], 'block_means')
        self.assertGreaterEqual(new['surround_blocks'], 4)

    def test_surround_override_other_fusion_modes_unchanged(self):
        from depth_image_lens.tuning_backend import MODES
        args, obj, ring = self.sparse_surround()
        for mode in MODES:
            if mode == 'color_first':
                continue
            with self.subTest(mode=mode):
                legacy = detect(*args, RGBDBackgroundSegmenter(fusion_mode=mode), work_roi=obj | ring, diagnostics=True)
                r = detect(*args, RGBDBackgroundSegmenter(fusion_mode=mode, first_min_surround_blocks=0), work_roi=obj | ring, diagnostics=True)
                np.testing.assert_equal(r.diagnostics, legacy.diagnostics)
                np.testing.assert_equal(r.object_mask, legacy.object_mask)
                self.assertEqual(r.stages, legacy.stages)

    def test_crop_not_eroded_as_boundary(self):
        c,z,bc,bz=scene();c[:24,10:34]=80;z[:24,10:34]=998
        r=self.run_first(c,z,bc,bz);self.assertTrue(r.object_mask[:24,10:34].all())
        self.assertTrue(r.diagnostics['first_interior_mask'][0,20])

    def test_two_touching_colors(self):
        c,z,bc,bz=scene();c=np.tile(c,(2,2,1));bc=np.tile(bc,(2,2,1));z=np.tile(z,(2,2));bz=np.tile(bz,(2,2));c[20:52,20:44]=(20,150,210);c[20:52,44:68]=(210,70,30);z[20:52,20:68]=998
        r=self.run_first(c,z,bc,bz);self.assertTrue(r.object_mask[20:52,20:68].all())
        self.assertEqual(len(r.stages['fusion']['regions']),2)

    def test_upper_exclusion(self):
        c,z,bc,bz=self.object(100)
        r=self.run_first(c,z,bc,bz,local_max_reference_separation_m=.05)
        self.assertFalse(r.object_mask.any())

    def test_diagnostics_no_effect_and_filter(self):
        args=self.object();s=RGBDBackgroundSegmenter(fusion_mode='color_first')
        a=detect(*args,s,diagnostics=True);b=detect(*args,s,diagnostics=False)
        np.testing.assert_array_equal(a.object_mask,b.object_mask)
        r=self.run_first(*args,min_object_area=1000)
        self.assertFalse(r.object_mask.any());self.assertFalse(r.inferred_foreground_mask.any())
        self.assertTrue(r.diagnostics['first_interior_mask'].any())

    def test_rejected_color_reference_has_no_depth_fallback(self):
        c,z,bc,bz=self.object(20)
        c[:]=20
        r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_first'),
                 background_check_mask=np.ones(z.shape,bool),diagnostics=True)
        self.assertIn(r.stages['color_reference']['status'],('rejected','insufficient'))
        self.assertEqual(r.stages['color_reference']['selection_action'],'disable_color_no_candidates')
        self.assertTrue(r.depth_foreground_mask.any())
        self.assertFalse(r.object_mask.any())

    def test_validation(self):
        for kw in ({'first_block_size_px':True},{'first_min_blocks':1},{'first_min_depth_delta_m':0},{'first_ring_gap_px':16},{'first_min_valid_fraction':0},{'first_noise_multiplier':float('nan')}):
            with self.assertRaises(ValueError):RGBDBackgroundSegmenter(**kw)
        for minimum in (False, -1, 10001, 1.0, '0', float('nan')):
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, 'first_min_surround_blocks'):
                RGBDBackgroundSegmenter(first_min_surround_blocks=minimum)

    def test_gui_and_debug(self):
        from depth_image_lens.tuning_backend import prepare_inputs,default_config,run_pipeline,diagnostic_metadata
        from .test_tuning_backend import scene as gui_scene
        config=default_config();config['diagnostics']=True;config['segmenter']['fusion_mode']='color_first'
        self.assertIsNone(config['segmenter']['first_min_surround_blocks'])
        config['segmenter']['first_min_surround_blocks']=0
        result=run_pipeline(prepare_inputs(gui_scene(),[40,40,80,80]),config)
        key='segmenter/first_candidate_mask';self.assertIn(key,result.images)
        self.assertIn('description',diagnostic_metadata(result,key))
        key='segmenter/first_failed_surround_reference_mask'
        self.assertIn(key,result.images)
        self.assertIn('description',diagnostic_metadata(result,key))
        r=self.run_first(*self.object());self.assertEqual(r.render_debug().ndim,3)
