"""Split-Lab metric compatibility, provenance and explicit failure cases."""
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from depth_image_lens._rgbd_color_model import _palette
from examples.example_rgbd_color_model_channels import scene,evaluate,fractions


class ColorModelChannelTests(unittest.TestCase):
    def test_equal_channels_preserve_default_results(self):
        data=scene(depth_noise_mm=2,seed=3)
        a=evaluate(data);b=evaluate(data,18,18)
        for key in ('object_mask','unknown_mask','suppressed_mask','deferred_mask'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        for key in ('model_all_rejection_flags','model_all_color_distances',
                    'model_all_selected_masks','model_background_distance'):
            np.testing.assert_array_equal(a.diagnostics[key],b.diagnostics[key])
        self.assertEqual(a.stages['fusion']['models'],b.stages['fusion']['models'])
        p=RGBDBackgroundSegmenter(model_color_tolerance=32).parameters
        self.assertEqual(p['model_luma_tolerance'],32)
        self.assertEqual(p['model_chroma_tolerance'],32)

    def test_luma_relaxation_restores_colored_edge_not_gray_shadow(self):
        self.assertEqual(fractions(evaluate(scene()))[1],0)
        r=evaluate(scene(),36,18)
        self.assertEqual(fractions(r),[1.,1.,0.])
        self.assertTrue(r.inferred_foreground_mask[12:52,40:48].all())
        self.assertTrue(r.unknown_mask[12:52,40:48].all())

    def test_gray_remains_distinguishable_and_shadow_leak_is_explicit(self):
        data=scene(core=(100,128,128),edge=(125,128,128))
        r=evaluate(data,36,18)
        self.assertEqual(fractions(r),[1.,1.,.25])
        # Ignoring brightness entirely would erase the only object/background cue.
        near=evaluate(scene(core=(100,128,128),edge=(148,128,128)),60,18)
        self.assertEqual(fractions(near)[1],0)

    def test_same_color_shadow_not_solved_by_channel_weights(self):
        r=evaluate(scene(shadow=(145,170,145)),36,18)
        self.assertEqual(fractions(r)[2],.25)

    def test_chroma_remains_independent(self):
        data=scene(edge=(110,195,145))
        self.assertEqual(fractions(evaluate(data,60,18))[1],0)
        self.assertEqual(fractions(evaluate(data,18,36))[1],1)

    def test_palette_training_uses_same_metric(self):
        colors=np.repeat(np.array([[80,160,130],[100,160,130]],float),10,axis=0)
        old=_palette(colors,6,18,.03)
        scaled=_palette(colors,6,18,.03,np.array([.3,1,1]))
        self.assertEqual(len(old),2);self.assertEqual(len(scaled),1)
        np.testing.assert_array_equal(scaled[0],[90,160,130])
        two_hues=np.repeat(np.array([[80,130,130],[80,170,130]],float),10,axis=0)
        self.assertEqual(len(_palette(two_hues,6,18,.03,np.array([.3,1,1]))),2)

    def test_diagnostic_components_belong_to_same_nearest_palette(self):
        r=evaluate(scene(depth_noise_mm=2,seed=5),60,18);d=r.diagnostics
        reconstructed=np.hypot(d['model_all_luma_distances']*.3,d['model_all_chroma_distances'])
        np.testing.assert_allclose(reconstructed,d['model_all_color_distances'],rtol=2e-6,atol=1e-5)
        bg=np.hypot(d['background_luma_distance']*.3,d['background_chroma_distance'])
        np.testing.assert_allclose(bg,d['model_background_distance'])
        ids=d['model_diagnostic_ids'];records={m['id']:m for m in r.stages['fusion']['models']}
        for i,model_id in enumerate(ids):
            indices=d['model_all_nearest_palette_indices'][i]
            self.assertTrue((indices[indices>=0]<len(records[int(model_id)]['colors_lab'])).all())
        f=d['model_all_rejection_flags']
        finite=np.isfinite(d['model_all_color_distances'])
        np.testing.assert_array_equal((f[finite]&64)!=0,d['model_all_color_distances'][finite]>18)
        expected=d['model_all_color_distances']+1>=bg
        np.testing.assert_array_equal((f[finite]&128)!=0,expected[finite])

    def test_no_depth_and_invalid_depth_not_fabricated(self):
        r=evaluate(scene(height_mm=0),60,18);self.assertFalse(r.object_mask.any())
        data=list(scene());data[1][:,40]=0
        r=evaluate(data,60,18)
        self.assertFalse(r.object_mask[:,40:].any())
        self.assertFalse(r.object_mask[:,40].any())

    def test_validation_and_other_modes_unchanged(self):
        for key in ('model_luma_tolerance','model_chroma_tolerance'):
            for value in (True,0,-1,np.nan,np.inf,442,.001,[18]):
                with self.assertRaises(ValueError):RGBDBackgroundSegmenter(**{key:value})
        with self.assertRaises(ValueError):
            RGBDBackgroundSegmenter(model_color_tolerance=.001,model_luma_tolerance=18)
        # Legacy tiny equal tolerance remains accepted when the new options are omitted.
        RGBDBackgroundSegmenter(model_color_tolerance=.001)
        self.assertEqual(RGBDBackgroundSegmenter().parameters['fusion_mode'],'union')

    def test_multiple_models_and_component_indices_are_consistent(self):
        import cv2
        data=list(scene());data[0][12:52,60:80]=[30,140,210];data[1][12:52,60:80]=.992
        r=evaluate(data,60,18);d=r.diagnostics
        self.assertGreaterEqual(len(d['model_diagnostic_ids']),2)
        lab=cv2.cvtColor(data[0],cv2.COLOR_RGB2LAB).astype(float)
        records={m['id']:m for m in r.stages['fusion']['models']}
        for i,model_id in enumerate(d['model_diagnostic_ids']):
            palette=np.array(records[int(model_id)]['colors_lab'])
            indices=d['model_all_nearest_palette_indices'][i]
            valid=indices>=0
            chosen=palette[indices[valid]]
            difference=lab[valid]-chosen
            np.testing.assert_allclose(np.abs(difference[:,0]),d['model_all_luma_distances'][i][valid])
            np.testing.assert_allclose(np.linalg.norm(difference[:,1:],axis=1),d['model_all_chroma_distances'][i][valid],rtol=1e-6)
            exact=np.stack([np.linalg.norm((lab-color)*[.3,1,1],axis=2) for color in palette])
            np.testing.assert_array_equal(np.argmin(exact,axis=0)[valid],indices[valid])

    def test_coherent_false_depth_seed_remains_a_failure_case(self):
        data=list(scene(height_mm=0))
        # A broad dark shadow with coherent depth bias is not distinguishable
        # from an object by this method. Color weights must not claim otherwise.
        data[1][12:52,48:56]=.992
        r=evaluate(data,60,18)
        self.assertGreater(fractions(r)[2],.9)

    def test_diagonal_channel_limits_form_ellipse_not_independent_box(self):
        r=evaluate(scene(edge=(140,190,145)),36,24)
        d=r.diagnostics;finite=np.isfinite(d['model_all_color_distances'])
        normalized=np.hypot(d['model_all_luma_distances']/36,d['model_all_chroma_distances']/24)
        np.testing.assert_allclose(d['model_all_color_distances'][finite]/18,normalized[finite],rtol=1e-6)
        both_under=(d['model_all_luma_distances']<36)&(d['model_all_chroma_distances']<24)
        outside=both_under&(normalized>1)
        self.assertTrue(outside.any())
        self.assertTrue(((d['model_all_rejection_flags'][outside]&64)!=0).all())

    def test_farther_policy_does_not_change_raw_positive_evidence(self):
        data=list(scene(edge=(110,170,145)));data[1][12:52,40:48]=1.003
        default=evaluate(data)
        loose=evaluate(data,model_farther_threshold_m=.004)
        off=evaluate(data,model_reject_farther=False)
        self.assertEqual(fractions(default)[1],0)
        self.assertEqual(fractions(loose)[1],1)
        self.assertEqual(fractions(off)[1],1)
        for r in (loose,off):
            np.testing.assert_array_equal(r.depth_foreground_mask,default.depth_foreground_mask)
            self.assertTrue(r.diagnostics['model_raw_farther_mask'][12:52,40:48].all())
            self.assertFalse(r.diagnostics['model_farther_excluded_mask'][12:52,40:48].any())
            self.assertTrue(r.diagnostics['model_accepted_raw_farther_mask'][12:52,40:48].all())
            self.assertTrue(r.inferred_foreground_mask[12:52,40:48].all())
            self.assertTrue(r.unknown_mask[12:52,40:48].all())
            self.assertFalse(r.confident_foreground_mask[12:52,40:48].any())
            self.assertFalse((r.diagnostics['model_all_rejection_flags'][:,12:52,40:48]&16).any())
        strict=evaluate(data,model_farther_threshold_m=.001)
        self.assertEqual(fractions(strict)[1],0)

    def test_disabling_farther_keeps_invalid_upper_roi_and_color_gates(self):
        data=list(scene());data[1][12:52,40:48]=1.003
        r=evaluate(data,model_reject_farther=False)
        self.assertEqual(fractions(r)[1],0)  # Different brightness still fails color.
        data=list(scene(edge=(110,170,145)));data[1][:,40]=0
        r=evaluate(data,model_reject_farther=False,model_background_margin=-100)
        self.assertFalse(r.object_mask[:,40:].any())
        data=list(scene(edge=(110,170,145)));data[1][12:52,40:48]=.7
        r=evaluate(data,model_reject_farther=False,local_max_reference_separation_m=.1)
        self.assertEqual(fractions(r)[1],0)
        self.assertTrue((r.diagnostics['model_all_rejection_flags'][:,12:52,40:48]&32).all())

    def test_negative_margin_can_admit_background_closer_than_model(self):
        data=scene(core=(100,128,128),edge=(135,128,128))
        normal=evaluate(data,60,18,model_background_margin=0)
        permissive=evaluate(data,60,18,model_background_margin=-4)
        self.assertEqual(fractions(normal)[1],0)
        self.assertEqual(fractions(permissive)[1],1)
        d=permissive.diagnostics
        self.assertTrue((d['model_all_color_distances'][:,20,43]>d['model_background_distance'][20,43]).all())
        self.assertTrue(permissive.inferred_foreground_mask[12:52,40:48].all())
        self.assertFalse(permissive.object_mask[0].any())

    def test_new_policy_validation(self):
        for value in (1,None,'off'):
            with self.assertRaises(ValueError):RGBDBackgroundSegmenter(model_reject_farther=value)
        for value in (True,0,-1,np.nan,np.inf,[.002],'bad'):
            with self.assertRaises(ValueError):RGBDBackgroundSegmenter(model_farther_threshold_m=value)
        RGBDBackgroundSegmenter(model_background_margin=-1)
        for value in (np.nan,np.inf,-np.inf,True):
            with self.assertRaises(ValueError):RGBDBackgroundSegmenter(model_background_margin=value)

    def test_color_local_ignores_model_policy_options(self):
        data=scene(depth_noise_mm=3,seed=6)
        a=evaluate(data,fusion_mode='color_local')
        b=evaluate(data,fusion_mode='color_local',model_reject_farther=False,
                   model_farther_threshold_m=.01,model_background_margin=-12)
        np.testing.assert_array_equal(a.object_mask,b.object_mask)
        np.testing.assert_array_equal(a.unknown_mask,b.unknown_mask)

    def test_area_filter_updates_negative_depth_recovery_provenance(self):
        data=list(scene(edge=(110,170,145)));data[1][12:52,40:48]=1.003
        r=evaluate(data,model_reject_farther=False,min_object_area=10000)
        self.assertFalse(r.object_mask.any())
        self.assertFalse(r.diagnostics['model_accepted_raw_farther_mask'].any())
        self.assertTrue(r.diagnostics['model_raw_farther_mask'].any())
