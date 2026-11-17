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

    def test_gui_and_debug(self):
        from depth_image_lens.tuning_backend import prepare_inputs,default_config,run_pipeline,diagnostic_metadata
        from .test_tuning_backend import scene as gui_scene
        config=default_config();config['diagnostics']=True;config['segmenter']['fusion_mode']='color_first'
        result=run_pipeline(prepare_inputs(gui_scene(),[40,40,80,80]),config)
        key='segmenter/first_candidate_mask';self.assertIn(key,result.images)
        self.assertIn('description',diagnostic_metadata(result,key))
        r=self.run_first(*self.object());self.assertEqual(r.render_debug().ndim,3)
