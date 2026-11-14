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

class ColorModelDiagnosticTests(unittest.TestCase):
    data = ColorModelTests.data
    run_model = ColorModelTests.run_model
    def test_diagnostics_never_change_selection(self):
        data=self.data()
        a=self.run_model(data)
        b=detect(*data,RGBDBackgroundSegmenter(fusion_mode='color_model'),diagnostics=False)
        for name in ('object_mask','unknown_mask','suppressed_mask','deferred_mask'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        self.assertFalse(b.diagnostics)

    def test_rejected_color_has_real_distance_and_overlapping_reasons(self):
        c,z,bc,bz=self.data();c[8:32,34:39]=(20,150,210)
        r=self.run_model((c,z,bc,bz),model_background_margin=200)
        d=r.diagnostics;flags=d['model_all_rejection_flags']
        self.assertTrue(np.isfinite(d['model_all_color_distances'][:,10,36]).all())
        self.assertTrue(((flags[:,10,36]&64)!=0).any())
        self.assertTrue(((flags[:,10,36]&128)!=0).any())
        self.assertTrue(((flags&256)!=0).all())
        self.assertTrue(np.isnan(d['model_all_source_distances_px']).all())
        self.assertTrue(np.isnan(d['model_color_distance'][10,36]))

    def test_connected_budget_and_disconnected_are_distinct(self):
        c,z,bc,bz=self.data();c[8:32,34:58]=80;c[35:39,40:55]=80
        r=self.run_model((c,z,bc,bz),model_boundary_radius_px=1)
        f=r.diagnostics['model_all_rejection_flags']
        self.assertTrue(((f[:,15,50]&1024)!=0).any())
        self.assertTrue(((f[:,37,45]&512)!=0).any())
        self.assertTrue(np.isfinite(r.diagnostics['model_all_source_distances_px'][:,37,45]).all())
        self.assertFalse(r.object_mask[15,50]);self.assertFalse(r.object_mask[37,45])

    def test_no_models_has_empty_stack_not_fake_infinite_distance(self):
        r=self.run_model(scene());d=r.diagnostics
        self.assertEqual(d['model_all_color_distances'].shape,(0,)+r.object_mask.shape)
        self.assertTrue(d['model_no_evaluable_model_mask'].all())
        self.assertGreater(r.render_debug().shape[0],0)

    def test_multiple_models_keep_separate_ids_and_selection(self):
        c,z,bc,bz=self.data();c[8:32,42:60]=(20,150,210);z[8:32,42:60]=980
        r=self.run_model((c,z,bc,bz));d=r.diagnostics
        self.assertGreaterEqual(len(d['model_diagnostic_ids']),2)
        np.testing.assert_array_equal(d['model_all_selected_masks'].any(axis=0),r.object_mask)
        self.assertTrue(((d['model_all_rejection_flags'][:,15,15]&64)!=0).any())
        self.assertTrue(r.object_mask[15,15])

    def test_input_exclusions_are_independent_and_color_invalid_is_nan(self):
        c,z,bc,bz=self.data();z[12,15]=0;z[12,16]=1020;z[12,17]=700
        roi=np.ones(z.shape,bool);roi[:,33:]=False
        color_valid=np.ones(z.shape,bool);color_valid[12,18]=False
        r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_model',
                 local_max_reference_separation_m=.1),work_roi=roi,
                 color_valid_mask=color_valid,diagnostics=True)
        d=r.diagnostics;f=d['model_all_rejection_flags']
        for y,x,bit in ((12,15,4),(12,16,16),(12,17,32),(12,18,2),(12,34,1)):
            self.assertTrue(((f[:,y,x]&bit)!=0).all())
            self.assertFalse(r.object_mask[y,x])
        self.assertTrue(np.isnan(d['model_all_color_distances'][:,12,18]).all())
        self.assertTrue(np.isfinite(d['model_all_color_distances'][:,12,15]).all())

    def test_spatial_tiles_keep_overlaps_and_model_identity(self):
        from depth_image_lens._rgbd_color_model import render_model_rejection_maps
        f=np.zeros((2,2,3),np.uint16)
        f[0,0,0]=64|128;f[1,0,1]=64;f[:,1,2]=64
        obj=np.zeros((2,3),bool);obj[1,2]=True
        d=dict(model_all_rejection_flags=f,model_diagnostic_ids=np.array([3,9]),
               model_no_evaluable_model_mask=np.zeros((2,3),bool))
        original=f.copy()
        both=render_model_rejection_maps(d,obj,900)
        one=render_model_rejection_maps(d,obj,900,model_id=3)
        # Tile 6 is bit 64, tile 7 is bit 128. Each image is 300x200.
        y=76+2*(200+56)+56
        np.testing.assert_array_equal(both[y+50,50],(255,125,35))
        np.testing.assert_array_equal(both[y+50,150],(255,125,35))
        np.testing.assert_array_equal(one[y+50,150],(0,0,0))
        np.testing.assert_array_equal(both[y+50,350],(255,125,35))
        np.testing.assert_array_equal(both[y+150,250],(70,70,70))
        np.testing.assert_array_equal(f,original)
        bgr=render_model_rejection_maps(d,obj,900,color_order='BGR')
        np.testing.assert_array_equal(bgr,both[...,::-1])
        for bad in (True,1.5,99):
            with self.assertRaises(ValueError):
                render_model_rejection_maps(d,obj,900,model_id=bad)

    def test_spatial_render_does_not_mutate_result_and_requires_diagnostics(self):
        r=self.run_model(self.data());before=r.object_mask.copy()
        original=r.diagnostics['model_all_rejection_flags'].copy()
        all_view=r.render_debug()
        one_view=r.render_debug(model_diagnostic_id=int(r.diagnostics['model_diagnostic_ids'][0]))
        self.assertEqual(all_view.shape,one_view.shape)
        np.testing.assert_array_equal(r.object_mask,before)
        np.testing.assert_array_equal(r.diagnostics['model_all_rejection_flags'],original)
        no_debug=detect(*self.data(),RGBDBackgroundSegmenter(fusion_mode='color_model'))
        with self.assertRaises(ValueError):no_debug.render_debug(model_diagnostic_id=1)
