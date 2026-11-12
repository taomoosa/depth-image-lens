"""Region-level hypotheses and counterexamples; not real-camera accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import scene, detect

class ColorRegionTests(unittest.TestCase):
    def run_region(self, data, **settings):
        return detect(*data, RGBDBackgroundSegmenter(fusion_mode='color_region', **settings), diagnostics=True)

    def data(self):
        c,z,bc,bz=scene();c[8:32,10:34]=120
        return c,z,bc,bz

    def test_checkerboard_depth_whole_region_not_square_support(self):
        c,z,bc,bz=self.data(); yy,xx=np.indices(z.shape)
        m=c[...,0]==120;z[m & ((yy+xx)%2==0)]=986
        r=self.run_region((c,z,bc,bz))
        np.testing.assert_array_equal(r.object_mask,m)
        self.assertTrue(r.unknown_mask[m & ~r.depth_foreground_mask].all())
        self.assertEqual(len(r.stages['fusion']['regions']),1)
        self.assertAlmostEqual(r.stages['fusion']['regions'][0]['depth_support_fraction'],.5)

    def test_no_square_depth_support_and_rgb_bgr(self):
        from unittest.mock import patch
        c,z,bc,bz=self.data();z[8:32,10:34]=994
        with patch('depth_image_lens._rgbd_fusion.depth_support', side_effect=AssertionError('square support called')):
            r=self.run_region((c,z,bc,bz))
        bgr=RGBDBackgroundSegmenter(fusion_mode='color_region').detect(
            c[...,::-1],z,background_color=bc[...,::-1],background_depth=bz,
            color_order='BGR',registered_to_depth=True,depth_scale_m=.001)
        np.testing.assert_array_equal(r.object_mask,bgr.object_mask)

    def test_noisy_shallow_aggregate(self):
        c,z,bc,bz=self.data();m=c[...,0]==120
        rng=np.random.default_rng(8);z=z.astype(float)
        z[m]=1000-rng.normal(7,5,m.sum())
        r=self.run_region((c,z,bc,bz))
        # Large farther errors are barriers; all remaining region pixels recover.
        self.assertGreater(r.object_mask[m].mean(),.95)

    def test_separate_same_color_cannot_borrow_support(self):
        c,z,bc,bz=self.data();z[8:32,10:34]=980
        c[8:20,40:55]=120
        r=self.run_region((c,z,bc,bz))
        self.assertFalse(r.object_mask[:,40:].any())
        self.assertEqual(len(r.stages['fusion']['regions']),2)

    def test_touching_different_color_shadow(self):
        c,z,bc,bz=self.data();z[8:32,10:34]=980;c[8:32,34:58]=(20,50,30)
        r=self.run_region((c,z,bc,bz));self.assertEqual(r.object_mask.sum(),576)

    def test_touching_same_color_shadow_is_known_failure(self):
        c,z,bc,bz=self.data();z[8:32,10:34]=980;c[8:32,34:46]=120
        r=self.run_region((c,z,bc,bz));self.assertTrue(r.object_mask[8:32,34:46].all())
        self.assertTrue(r.inferred_foreground_mask[8:32,34:46].all())

    def test_color_hole_preserved_same_color_hole_inferred(self):
        for color_hole in (False, True):
            c,z,bc,bz=self.data();z[8:32,10:34]=980;z[16:20,18:22]=1000
            if color_hole:c[16:20,18:22]=140
            r=self.run_region((c,z,bc,bz))
            self.assertEqual(bool(r.object_mask[16:20,18:22].all()),not color_hole)

    def test_isolated_enormous_outlier_does_not_dominate_mean(self):
        c,z,bc,bz=self.data();z[12,12]=100
        self.assertFalse(self.run_region((c,z,bc,bz)).object_mask.any())

    def test_density_and_mean_both_required(self):
        c,z,bc,bz=self.data();z[8:32,10:20]=980
        self.assertTrue(self.run_region((c,z,bc,bz)).object_mask.any())
        self.assertFalse(self.run_region((c,z,bc,bz),region_min_depth_fraction=.7).object_mask.any())
        z[8:32,10:34]=996
        self.assertFalse(self.run_region((c,z,bc,bz)).object_mask.any())

    def test_missing_depth_not_filled_coverage_gate(self):
        c,z,bc,bz=self.data();z[8:32,10:34]=980;z[8:20,10:34]=0
        r=self.run_region((c,z,bc,bz));self.assertEqual(r.object_mask.sum(),288)
        self.assertTrue(r.unknown_mask[8:20,10:34].all())
        self.assertFalse(self.run_region((c,z,bc,bz),region_min_valid_fraction=.75).object_mask.any())

    def test_farther_depth_votes_negative_and_is_never_selected(self):
        c,z,bc,bz=self.data();z[8:32,10:20]=980;z[:,20]=1020
        r=self.run_region((c,z,bc,bz));self.assertFalse(r.object_mask[:,20].any())

    def test_zero_mean_depth_noise_not_positive_censored(self):
        for sigma in (10,15,20):
            c=np.full((120,160,3),140,np.uint8);bc=c.copy()
            c[10:110,10:150]=100
            bz=np.full((120,160),1000.)
            z=bz.copy();m=c[...,0]==100
            z[m]=1000-np.random.default_rng(0).normal(0,sigma,m.sum())
            r=self.run_region((c,z,bc,bz))
            self.assertFalse(r.object_mask.any())
            self.assertEqual(len(r.stages['fusion']['regions']),1)
            self.assertLess(abs(r.stages['fusion']['regions'][0]['clipped_mean_depth_ratio']),.1)

    def test_roi_crop_and_empty(self):
        c,z,bc,bz=scene()
        self.assertFalse(self.run_region((c,z,bc,bz)).object_mask.any())
        c[:10,:10]=120;z[:10,:10]=980
        r=self.run_region((c,z,bc,bz));self.assertEqual(r.object_mask.sum(),100)

    def test_small_and_color_budget(self):
        c,z,bc,bz=self.data();z[8:32,10:34]=980
        self.assertFalse(self.run_region((c,z,bc,bz),region_min_area=600).object_mask.any())
        c[8:32,34:58]=(200,40,80);z[8:32,34:58]=980
        r=self.run_region((c,z,bc,bz),region_max_color_groups=1)
        self.assertTrue(r.metrics['color_region_budget_exceeded'])
        self.assertTrue(r.unknown_mask[(c[...,0]!=140)&~r.object_mask].all())

    def test_json_debug_inputs_unchanged(self):
        data=self.data();data[1][8:32,10:34]=980; copies=[x.copy() for x in data]
        r=self.run_region(data);json.dumps(r.to_dict(),allow_nan=False)
        self.assertEqual(r.render_debug(data[0]).dtype,np.uint8)
        for x,y in zip(data,copies):np.testing.assert_array_equal(x,y)

    def test_parameter_validation(self):
        for kwargs in ({'region_color_tolerance':0},{'region_min_valid_fraction':0},
                       {'region_min_area':True},{'region_max_color_groups':257},
                       {'region_mean_depth_ratio':3},{'region_depth_clip_ratio':float('nan')}):
            with self.assertRaises(ValueError):RGBDBackgroundSegmenter(fusion_mode='color_region',**kwargs)

    def test_coherent_depth_error_is_known_failure(self):
        data=self.data();data[1][8:32,10:34]=980
        self.assertEqual(self.run_region(data).object_mask.sum(),576)
