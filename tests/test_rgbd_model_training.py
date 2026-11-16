"""Training-only guard contracts, including cases where evidence is insufficient."""
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import scene, detect

class ColorCoreTrainingTests(unittest.TestCase):
    def run_guard(self, c, z, bc, bz, **kw):
        return detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_model',
                      model_training_policy='color_core', **kw), diagnostics=True)

    def test_legacy_is_default_and_opt_in_validation(self):
        self.assertEqual(RGBDBackgroundSegmenter().parameters['model_training_policy'], 'legacy')
        for kw in ({'model_training_policy':'bad'}, {'model_training_inset_px':-1},
                   {'model_training_inset_px':True}, {'model_training_inset_px':11}):
            with self.assertRaises(ValueError): RGBDBackgroundSegmenter(**kw)

    def test_inset_only_changes_learning_not_output_boundary(self):
        c,z,bc,bz=scene(); c[8:32,10:34]=80; z[8:32,10:34]=980
        r=self.run_guard(c,z,bc,bz)
        self.assertTrue(r.object_mask[8:32,10:34].all())
        self.assertFalse(r.diagnostics['model_training_used_mask'][8,15])
        self.assertTrue(r.diagnostics['model_training_input_mask'][8,15])
        self.assertTrue(r.diagnostics['model_training_used_mask'][15,15])

    def test_two_touching_colors_retained(self):
        c,z,bc,bz=scene();c[8:32,10:22]=(20,150,210);c[8:32,22:34]=(210,70,30);z[8:32,10:34]=980
        r=self.run_guard(c,z,bc,bz,model_max_colors=2)
        self.assertTrue(r.object_mask[8:32,10:34].all())

    def test_no_fallback_for_narrow_color_without_interior(self):
        c,z,bc,bz=scene();c[8:32,20:23]=80;z[8:32,20:23]=980
        r=self.run_guard(c,z,bc,bz)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.diagnostics['model_training_no_color_interior_mask'].any())
        self.assertTrue(r.unknown_mask[8:32,20:23].all())

    def test_crop_edge_is_not_a_color_boundary(self):
        c,z,bc,bz=scene();c[:24,10:34]=80;z[:24,10:34]=980
        r=self.run_guard(c,z,bc,bz)
        self.assertTrue(r.diagnostics['model_training_used_mask'][0,20])
        self.assertTrue(r.object_mask[:24,10:34].all())

    def test_diagnostic_on_off_and_actual_training_not_final_selection(self):
        c,z,bc,bz=scene();c[8:32,10:34]=80;z[8:32,10:34]=980
        s=RGBDBackgroundSegmenter(fusion_mode='color_model',model_training_policy='color_core',min_object_area=1000)
        a=detect(c,z,bc,bz,s,diagnostics=True); b=detect(c,z,bc,bz,s,diagnostics=False)
        np.testing.assert_array_equal(a.object_mask,b.object_mask)
        self.assertFalse(a.diagnostics['model_training_mask'].any())
        self.assertTrue(a.diagnostics['model_training_input_mask'].any())
        self.assertTrue(a.diagnostics['model_training_used_mask'].any())

    def test_depth_noise_band_does_not_validate_whole_table_color(self):
        c,z,bc,bz=scene();c[8:32,10:30]=80;z[8:32,10:30]=980
        c[:,30:]=120;z[8:32,30:32]=980
        r=self.run_guard(c,z,bc,bz,model_max_colors=2,local_color_relaxation_ratio=.01)
        self.assertTrue(r.object_mask[8:32,10:30].all())
        self.assertFalse(r.object_mask[:,30:].any())
        self.assertTrue(r.diagnostics['model_training_excluded_mask'][:,30:].any())

    def test_gui_exposes_training_images_and_metadata(self):
        from depth_image_lens.tuning_backend import prepare_inputs, default_config, run_pipeline, diagnostic_metadata
        from .test_tuning_backend import scene as gui_scene
        config=default_config();config['diagnostics']=True
        config['segmenter'].update(fusion_mode='color_model', model_training_policy='color_core')
        result=run_pipeline(prepare_inputs(gui_scene(),[40,40,80,80]), config)
        for name in ('model_training_input_mask', 'model_training_used_mask', 'model_training_interior_mask'):
            key='segmenter/'+name
            self.assertIn(key,result.images)
            self.assertIn('description',diagnostic_metadata(result,key))
