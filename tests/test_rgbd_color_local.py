"""Synthetic local evidence contracts, not real-camera accuracy measurements."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import detect, scene

class ColorLocalTests(unittest.TestCase):
    def run_local(self, data, **settings):
        return detect(*data, RGBDBackgroundSegmenter(fusion_mode='color_local', **settings), diagnostics=True)

    def gray(self, lift=20):
        c,z,bc,bz=scene();c[8:32,10:34]=80;z[8:32,10:34]=1000-lift
        return c,z,bc,bz

    def test_empty_and_dark_shadow_no_depth(self):
        self.assertFalse(self.run_local(scene()).object_mask.any())
        r=self.run_local(self.gray(0));self.assertFalse(r.object_mask.any())
        self.assertTrue(r.diagnostics['local_background_mask'][12:28,14:30].all())

    def test_gray_object_without_chroma_seed(self):
        r=self.run_local(self.gray());self.assertTrue(r.object_mask[8:32,10:34].all())

    def test_weak_color_noisy_thin_object(self):
        c,z,bc,bz=self.gray();c[8:32,10:34]=125;z=z.astype(float)
        z[8:32,10:34]=1000-np.random.default_rng(2).normal(3,4,(24,24))
        r=self.run_local((c,z,bc,bz));self.assertGreater(r.object_mask[8:32,10:34].mean(),.85)
        inferred=r.object_mask & ~r.depth_foreground_mask
        self.assertTrue(inferred.any());self.assertTrue(r.inferred_foreground_mask[inferred].all())
        self.assertTrue(r.unknown_mask[inferred].all())

    def test_sparse_checkerboard_support(self):
        c,z,bc,bz=self.gray(0);yy,xx=np.indices(z.shape);m=c[...,0]==80
        z[m & ((xx+yy)%3==0)]=990
        r=self.run_local((c,z,bc,bz));self.assertGreater(r.object_mask[m].mean(),.95)

    def test_touching_same_color_shadow_is_local_not_whole_region(self):
        c,z,bc,bz=self.gray();c[8:32,34:58]=80
        r=self.run_local((c,z,bc,bz));self.assertTrue(r.object_mask[8:32,10:34].all())
        self.assertFalse(r.object_mask[:,42:].any())
        # Some boundary leakage remains fundamentally possible.
        self.assertTrue(r.object_mask[12:28,34:36].any())

    def test_far_different_color_shadow_does_not_borrow_support(self):
        c,z,bc,bz=self.gray();c[8:32,34:58]=(20,150,200)
        r=self.run_local((c,z,bc,bz));self.assertFalse(r.object_mask[:,34:].any())

    def test_isolated_outlier_not_a_seed(self):
        c,z,bc,bz=self.gray(0);z[16,20]=100
        self.assertFalse(self.run_local((c,z,bc,bz)).object_mask.any())

    def test_missing_depth_never_filled_both_policies(self):
        c,z,bc,bz=self.gray();z[16:20,18:22]=0
        for policy in ('color','unknown'):
            r=self.run_local((c,z,bc,bz),missing_depth_policy=policy)
            self.assertFalse(r.object_mask[16:20,18:22].any())
            self.assertTrue(r.unknown_mask[16:20,18:22].all())

    def test_valid_fraction_gate(self):
        c,z,bc,bz=self.gray();z[8:32,10:34]=0;z[20,20]=980
        r=self.run_local((c,z,bc,bz));self.assertFalse(r.object_mask.any())

    def test_farther_is_barrier(self):
        c,z,bc,bz=self.gray();z[:,20]=1020
        self.assertFalse(self.run_local((c,z,bc,bz)).object_mask[:,20].any())

    def test_upper_band_excluded_from_support(self):
        c,z,bc,bz=self.gray();z[8:32,10:34]=700
        r=self.run_local((c,z,bc,bz),local_max_reference_separation_m=.1)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.diagnostics['local_upper_excluded_mask'][8:32,10:34].all())

    def test_roi_never_leaks(self):
        data=self.gray();roi=np.zeros(data[1].shape,bool);roi[:,:20]=1
        r=detect(*data,RGBDBackgroundSegmenter(fusion_mode='color_local'),work_roi=roi,diagnostics=True)
        self.assertFalse(r.object_mask[~roi].any());self.assertFalse(r.unknown_mask[~roi].any())

    def test_area_filter_updates_provenance(self):
        r=self.run_local(self.gray(),min_object_area=1000)
        self.assertFalse(r.object_mask.any());self.assertFalse(r.diagnostics['local_seed_mask'].any())
        self.assertFalse(r.confident_foreground_mask.any());self.assertFalse(r.inferred_foreground_mask.any())

    def test_bgr_depth_units_and_no_mutation(self):
        data=self.gray();copies=[a.copy() for a in data];r=self.run_local(data)
        c,z,bc,bz=data
        other=RGBDBackgroundSegmenter(fusion_mode='color_local').detect(c[...,::-1],z/1000,background_color=bc[...,::-1],background_depth=bz/1000,color_order='BGR',registered_to_depth=True,depth_scale_m=1)
        np.testing.assert_array_equal(r.object_mask,other.object_mask)
        for a,b in zip(data,copies):np.testing.assert_array_equal(a,b)
        json.dumps(r.to_dict(),allow_nan=False);self.assertEqual(r.render_debug(c).dtype,np.uint8)

    def test_coherent_bias_is_known_failure(self):
        # Numerically identical to an object; no claim of bias robustness.
        self.assertTrue(self.run_local(self.gray(6)).object_mask.any())

    def test_noiseless_subthreshold_thin_object_is_known_limitation(self):
        # No samples exceed depth_threshold: lowering mean alone cannot recover.
        self.assertFalse(self.run_local(self.gray(3)).object_mask.any())
        self.assertTrue(self.run_local(self.gray(3),depth_threshold_m=.002,depth_margin_m=.0005).object_mask.any())

    def test_color_uncertainty_is_preserved(self):
        r=self.run_local(self.gray())
        self.assertTrue(r.uncertain_mask[8:32,10:34].all())
        self.assertTrue(r.unknown_mask[8:32,10:34].all())
        self.assertFalse(r.confident_foreground_mask[8:32,10:34].any())

    def test_depth_threshold_margin_is_preserved(self):
        c,z,bc,bz=self.gray(6);c[8:32,10:34]=(30,150,210)
        r=self.run_local((c,z,bc,bz),depth_margin_m=.0011)
        self.assertTrue(r.object_mask[8:32,10:34].all())
        self.assertTrue(r.uncertain_mask[8:32,10:34].all())
        self.assertTrue(r.unknown_mask[8:32,10:34].all())
        self.assertFalse(r.confident_foreground_mask[8:32,10:34].any())

    def test_correlated_zero_mean_depth_error_is_known_failure(self):
        c,z,bc,bz=self.gray(0)
        z[8:20,10:34]=994;z[20:32,10:34]=1006
        r=self.run_local((c,z,bc,bz))
        # Whole-region signed mean is zero, but a coherent positive patch can
        # still be mistaken for an object. Do not claim local bias immunity.
        self.assertTrue(r.object_mask[10:16,14:30].all())

    def test_upper_invalid_and_roi_cannot_supply_seeds(self):
        c,z,bc,bz=self.gray(0);z[8:32,10:34]=700;z[16:20,18:22]=0
        r=self.run_local((c,z,bc,bz),local_max_reference_separation_m=.1)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.unknown_mask[8:32,10:34].all())

    def test_validation(self):
        for kw in ({'local_support_radius_px':0},{'local_support_radius_px':True},
                   {'local_growth_radius_px':21},{'local_seed_min_area':0},
                   {'local_mean_depth_m':0},{'local_color_sigma':0},
                   {'local_background_depth_m':.002},{'local_depth_clip_m':.001},
                   {'local_min_valid_fraction':1.1},{'local_min_depth_fraction':0},
                   {'local_background_max_depth_fraction':.18},
                   {'local_color_relaxation_ratio':float('nan')},
                   {'local_max_reference_separation_m':-1},{'local_max_reference_separation_m':'bad'}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):RGBDBackgroundSegmenter(fusion_mode='color_local',**kw)
