"""Fixed color-model inference contracts and explicit synthetic counterexamples."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import scene, detect

class ColorModelTests(unittest.TestCase):
    def data(self):
        c,z,bc,bz=scene();c[8:32,10:34]=80;z[8:32,10:34]=980
        return c,z,bc,bz

    def run_model(self,data,**kw):
        return detect(*data,RGBDBackgroundSegmenter(fusion_mode='color_model',**kw),diagnostics=True)

    def test_empty_and_shadow_have_no_model(self):
        self.assertFalse(self.run_model(scene()).object_mask.any())
        c,z,bc,bz=self.data();z[:]=1000
        self.assertFalse(self.run_model((c,z,bc,bz)).object_mask.any())

    def test_no_depth_training_no_color_only_model(self):
        c,z,bc,bz=self.data();z[:]=997
        r=self.run_model((c,z,bc,bz));self.assertFalse(r.object_mask.any())

    def test_table_like_edge_rescue_is_inferred(self):
        c,z,bc,bz=self.data();c[8:32,34:39]=80
        r=self.run_model((c,z,bc,bz));self.assertTrue(r.object_mask[12:28,34:39].all())
        rescued=r.diagnostics['model_rescued_background_mask'];self.assertTrue(rescued.any())
        self.assertTrue(r.inferred_foreground_mask[rescued].all());self.assertTrue(r.unknown_mask[rescued].all())
        self.assertFalse(r.confident_foreground_mask[rescued].any())

    def test_growth_is_bounded(self):
        c,z,bc,bz=self.data();c[8:32,34:58]=80
        r=self.run_model((c,z,bc,bz),model_boundary_radius_px=2)
        # Local statistical seeds can extend a few pixels themselves; inference
        # distance is measured from actual source mask, not ground-truth edge.
        source=r.diagnostics['model_source_mask'];ys,xs=np.nonzero(source)
        self.assertFalse(r.object_mask[:,xs.max()+3:].any())

    def test_background_competition_blocks_background_like_color(self):
        c,z,bc,bz=self.data();c[8:32,10:34]=135
        r=self.run_model((c,z,bc,bz),local_color_relaxation_ratio=.01,model_background_margin=20)
        self.assertFalse(r.object_mask.any())

    def test_color_difference_alone_is_not_sufficient(self):
        c,z,bc,bz=self.data();c[8:32,34:58]=(20,150,210)
        r=self.run_model((c,z,bc,bz));self.assertFalse(r.object_mask[:,34:].any())

    def test_disconnected_matching_color_cannot_borrow_model(self):
        c,z,bc,bz=self.data();c[8:32,40:55]=80
        r=self.run_model((c,z,bc,bz));self.assertFalse(r.object_mask[:,40:].any())

    def test_multiple_colors_not_averaged_into_one(self):
        c,z,bc,bz=self.data();c[8:32,10:22]=(20,150,210);c[8:32,22:34]=(210,70,30)
        r=self.run_model((c,z,bc,bz));self.assertTrue(r.object_mask[8:32,10:34].all())
        colors=sum(len(m['colors_lab']) for m in r.stages['fusion']['models'])
        self.assertGreaterEqual(colors,2)

    def test_model_does_not_learn_from_inferred_color(self):
        c,z,bc,bz=self.data();c[8:32,34:38]=85;c[8:32,38:42]=95;c[8:32,42:46]=110
        r=self.run_model((c,z,bc,bz),model_boundary_radius_px=20,model_color_tolerance=10)
        self.assertFalse(r.object_mask[:,42:].any())

    def test_missing_depth_and_color_are_never_filled(self):
        c,z,bc,bz=self.data();z[15:18,18:21]=0
        r=self.run_model((c,z,bc,bz));self.assertFalse(r.object_mask[15:18,18:21].any())
        cv=np.ones(z.shape,bool);cv[20:23,22:25]=False
        r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_model'),color_valid_mask=cv,diagnostics=True)
        self.assertFalse(r.object_mask[~cv].any())

    def test_farther_upper_and_roi_protection(self):
        c,z,bc,bz=self.data();z[:,20]=1020;z[10:15,25:30]=700
        roi=np.ones(z.shape,bool);roi[:,32:]=False
        r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_model',local_max_reference_separation_m=.1),work_roi=roi,diagnostics=True)
        self.assertFalse(r.object_mask[:,20].any());self.assertFalse(r.object_mask[10:15,25:30].any())
        self.assertFalse(r.object_mask[~roi].any())

    def test_training_area_gate_and_budget(self):
        r=self.run_model(self.data(),model_min_training_pixels=1000)
        self.assertFalse(r.object_mask.any());self.assertEqual(r.stages['fusion']['models'][0]['status'],'insufficient_depth_training')
        self.assertTrue((r.diagnostics['fusion_decision'][8:32,10:34]==5).all())
        c,z,bc,bz=self.data();c[8:32,42:55]=80;z[8:32,42:55]=980
        r=self.run_model((c,z,bc,bz),model_max_regions=1)
        self.assertTrue(r.metrics['model_budget_exceeded']);self.assertFalse(r.object_mask[:,42:].any())

    def test_seed_filter_preserved(self):
        r=self.run_model(self.data(),local_seed_min_area=1000)
        self.assertFalse(r.object_mask.any())

    def test_area_filter_updates_provenance(self):
        r=self.run_model(self.data(),min_object_area=1000)
        self.assertFalse(r.object_mask.any());self.assertFalse(r.diagnostics['model_training_mask'].any())
        self.assertFalse(r.inferred_foreground_mask.any())

    def test_color_uncertainty_is_preserved(self):
        r=self.run_model(self.data());self.assertTrue(r.uncertain_mask[8:32,10:34].all())
        self.assertFalse(r.confident_foreground_mask[8:32,10:34].any())

    def test_coherent_large_false_seed_is_known_failure(self):
        # A shadow with consistent biased depth is indistinguishable from a real
        # depth-supported object. Area threshold 256 cannot reject this scene.
        r=self.run_model(self.data(),local_seed_min_area=256)
        self.assertTrue(r.object_mask[8:32,10:34].all())

    def test_same_color_shadow_rescue_is_known_failure(self):
        c,z,bc,bz=self.data();c[8:32,34:39]=80
        self.assertTrue(self.run_model((c,z,bc,bz)).object_mask[12:28,34:39].all())

    def test_json_debug_determinism_and_no_mutation(self):
        data=self.data();copies=[a.copy() for a in data];r=self.run_model(data);q=self.run_model(data)
        np.testing.assert_array_equal(r.object_mask,q.object_mask)
        json.dumps(r.to_dict(),allow_nan=False);self.assertEqual(r.render_debug(data[0]).dtype,np.uint8)
        for a,b in zip(data,copies):np.testing.assert_array_equal(a,b)

    def test_units_bgr_and_crop_edge(self):
        c,z,bc,bz=self.data();c[:8,10:34]=80;z[:8,10:34]=980
        r=self.run_model((c,z,bc,bz))
        q=RGBDBackgroundSegmenter(fusion_mode='color_model').detect(
            c[...,::-1],z/1000,background_color=bc[...,::-1],background_depth=bz/1000,
            color_order='BGR',registered_to_depth=True,depth_scale_m=1,diagnostics=True)
        np.testing.assert_array_equal(r.object_mask,q.object_mask)
        self.assertTrue(r.object_mask[0,10:34].all())

    def test_narrow_color_bridge_and_invalid_cut(self):
        c,z,bc,bz=self.data();c[19:21,34:40]=80
        r=self.run_model((c,z,bc,bz));self.assertTrue(r.object_mask[19:21,34:40].all())
        z[19:21,35]=0
        r=self.run_model((c,z,bc,bz));self.assertFalse(r.object_mask[19:21,36:40].any())

    def test_validation(self):
        for kw in ({'model_color_tolerance':0},{'model_background_margin':-1},
                   {'model_boundary_radius_px':31},{'model_min_training_pixels':2},
                   {'model_max_colors':0},{'model_max_regions':0},
                   {'model_min_color_fraction':0}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):RGBDBackgroundSegmenter(fusion_mode='color_model',**kw)
