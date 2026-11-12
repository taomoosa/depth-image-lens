"""Synthetic color-supported contracts and failures, not real sensor accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from .test_rgbd_background import scene, detect


class ColorSupportedTests(unittest.TestCase):
    def run_mode(self, data, settings=None, **kwargs):
        return detect(*data, RGBDBackgroundSegmenter(
            fusion_mode='color_supported', **(settings or {})), **kwargs)

    def gray(self, border=0):
        c, z, bc, bz = scene()
        c[8:32, 10:34] = 120
        z[8+border:32-border, 10+border:34-border] = 980
        return c, z, bc, bz

    def test_gray_seed_and_weak_edges_recovered(self):
        for border in (0, 3):
            data = self.gray(border)
            r = self.run_mode(data, diagnostics=True)
            np.testing.assert_array_equal(r.object_mask, data[0][..., 0] == 120)
            self.assertFalse(r.color_foreground_mask.any())
            self.assertTrue(r.diagnostics['depth_seed_mask'].any())
            np.testing.assert_array_equal(r.depth_foreground_mask, data[1] < data[3])
            self.assertTrue(r.unknown_mask[r.inferred_foreground_mask].all())
            self.assertFalse(r.confident_foreground_mask.any())

    def test_growth_bounded_and_zero_disables_it(self):
        data = self.gray(5)
        for radius in (0, 2, 4):
            r = self.run_mode(data, {'supported_growth_radius_px': radius}, diagnostics=True)
            self.assertLessEqual(r.diagnostics['growth_step'].max(), radius)
            self.assertFalse(r.object_mask[8, 10])
        r = self.run_mode(self.gray(3), {'supported_growth_radius_px': 0})
        np.testing.assert_array_equal(r.object_mask, r.depth_foreground_mask)

    def test_separate_and_touching_shadow_not_global_component_vote(self):
        for touch in (False, True):
            c, z, bc, bz = scene()
            bc[:] = (80,160,100); c[:] = bc
            obj = np.zeros(z.shape,bool); obj[10:30, 5:25] = True
            shadow = np.zeros(z.shape,bool)
            shadow[10:30, 25 if touch else 30:45 if touch else 50] = True
            c[obj] = (170,50,120); z[obj] = 980
            c[shadow] = (8,16,10)
            r = self.run_mode((c,z,bc,bz), diagnostics=True)
            np.testing.assert_array_equal(r.object_mask, obj)
            self.assertTrue(r.deferred_mask[shadow].all())
            self.assertFalse(r.diagnostics['accepted_color_only_mask'].any())
            self.assertEqual(int(r.diagnostics['fusion_decision'][shadow][0]), 3)

    def test_brightness_only_hole_with_background_color_is_preserved(self):
        c,z,bc,bz = self.gray()
        c[15:20,17:22] = bc[15:20,17:22]
        z[15:20,17:22] = 1000
        r = self.run_mode((c,z,bc,bz))
        np.testing.assert_array_equal(r.object_mask, z < bz)

    def test_single_spike_and_tiny_seed_suppression(self):
        c,z,bc,bz = scene(); c[10:20,10:20]=(30,180,220); z[12,12]=900
        r=self.run_mode((c,z,bc,bz))
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.depth_foreground_mask[12,12])
        self.assertTrue(r.suppressed_mask[12,12])

    def test_multiple_colors_keep_separate_prototypes(self):
        c,z,bc,bz=scene()
        c[10:30,5:25]=(220,40,40); c[10:30,25:45]=(40,40,220)
        z[13:27,8:22]=980; z[13:27,28:42]=980
        r=self.run_mode((c,z,bc,bz))
        expected=np.zeros(z.shape,bool);expected[10:30,5:45]=True
        np.testing.assert_array_equal(r.object_mask,expected)

    def test_connected_same_color_shadow_is_known_limit(self):
        c,z,bc,bz=self.gray()
        c[8:32,34:40]=120  # indistinguishable from a mixed-depth edge
        r=self.run_mode((c,z,bc,bz))
        self.assertTrue(r.object_mask[8:32,34:38].all())
        self.assertFalse(r.object_mask[:,38:40].any())
        self.assertTrue(r.unknown_mask[8:32,34:38].all())

    def test_coherent_false_depth_shadow_is_known_limit(self):
        c,z,bc,bz=scene();c[10:30,10:30]=(20,50,30);z[10:30,10:30]=980
        r=self.run_mode((c,z,bc,bz))
        self.assertEqual(r.object_mask.sum(),400)
        self.assertEqual(r.confident_foreground_mask.sum(),400)

    def test_same_color_hole_can_be_inferred_known_limit(self):
        c,z,bc,bz=self.gray();z[16:18,18:20]=1000
        r=self.run_mode((c,z,bc,bz))
        self.assertTrue(r.object_mask[16:18,18:20].all())
        self.assertTrue(r.inferred_foreground_mask[16:18,18:20].all())
        self.assertTrue(r.unknown_mask[16:18,18:20].all())

    def test_shallow_unseeded_gray_is_known_limit(self):
        c,z,bc,bz=self.gray();z[z<1000]=996
        r=self.run_mode((c,z,bc,bz))
        self.assertFalse(r.object_mask.any())
        tuned=self.run_mode((c,z,bc,bz), {'depth_threshold_m':.002,'depth_margin_m':.0005,
                                        'supported_anchor_ratio':1.5})
        self.assertEqual(tuned.object_mask.sum(),576)

    def test_support_fraction_is_veto_not_unlimited_region_acceptance(self):
        r=self.run_mode(self.gray(3), {'supported_min_depth_fraction':.9}, diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertTrue(r.diagnostics['support_rejected_mask'].any())
        self.assertFalse(r.diagnostics['supported_seed_mask'].any())
        self.assertTrue(r.unknown_mask[r.diagnostics['support_rejected_mask']].all())

    def test_missing_and_farther_depth_block_growth(self):
        c,z,bc,bz=self.gray()
        z[16:20,17:20]=0;z[24:27,17:20]=1100
        for policy in ('unknown','color'):
            r=self.run_mode((c,z,bc,bz),{'missing_depth_policy':policy})
            self.assertFalse(r.object_mask[(z==0)|(z==1100)].any())
            self.assertTrue(r.unknown_mask[(z==0)|(z==1100)].all())
            self.assertFalse(r.observed_mask[z==0].any())

    def test_invalid_color_and_roi_no_filling(self):
        data=self.gray();valid=np.ones(data[1].shape,bool);valid[12:20,14:20]=False
        r=self.run_mode(data,color_valid_mask=valid)
        self.assertFalse(r.object_mask[~valid].any())
        roi=np.zeros_like(valid);roi[10:25,10:25]=True
        r=self.run_mode(data,work_roi=roi)
        self.assertFalse(r.object_mask[~roi].any())
        self.assertFalse(r.unknown_mask[~roi].any())

    def test_rejected_reference_keeps_depth_seeds_not_color_expansion(self):
        c,z,bc,bz=self.gray(3)
        check=np.zeros(z.shape,bool);check[:5]=True;c[:5]=240
        r=self.run_mode((c,z,bc,bz),background_check_mask=check, diagnostics=True)
        self.assertEqual(r.stages['color_reference']['status'],'rejected')
        np.testing.assert_array_equal(r.object_mask,z<bz)
        self.assertTrue(r.unknown_mask[r.object_mask].all())

    def test_rgb_bgr_units_and_inputs_unchanged(self):
        data=self.gray(3);copies=[x.copy() for x in data]
        r=self.run_mode(data,diagnostics=True)
        c,z,bc,bz=data
        rb=RGBDBackgroundSegmenter(fusion_mode='color_supported').detect(
            c[...,::-1],z.astype(float)*.001,background_color=bc[...,::-1],
            background_depth=bz.astype(float)*.001,color_order='BGR',
            registered_to_depth=True,depth_scale_m=1.)
        np.testing.assert_array_equal(r.object_mask,rb.object_mask)
        for x,y in zip(data,copies):np.testing.assert_array_equal(x,y)
        json.dumps(r.to_dict(),allow_nan=False)
        image=r.render_debug(c)
        self.assertEqual(image.dtype,np.uint8)
        self.assertEqual(image.shape[2],3)

    def test_area_filter_preserves_unknown(self):
        r=self.run_mode(self.gray(), {'min_object_area':1000}, diagnostics=True)
        self.assertFalse(r.object_mask.any())
        self.assertFalse(r.inferred_foreground_mask.any())
        self.assertFalse(r.confident_foreground_mask.any())
        self.assertFalse(r.diagnostics['supported_seed_mask'].any())
        self.assertTrue(r.unknown_mask[8:32,10:34].all())

    def test_bad_parameters(self):
        values={'supported_color_tolerance':(0,-1,442,np.nan,np.inf,True),
                'supported_growth_radius_px':(-1,21,1.2,True),
                'supported_min_seed_area':(0,-1,1.2,True),
                'supported_min_depth_fraction':(0,1.1,np.nan,True),
                'supported_anchor_ratio':(.9,4.1,np.nan,True),
                'supported_relaxation_ratio':(0,1.1,np.nan,True)}
        for key,items in values.items():
            for value in items:
                with self.subTest(key=key,value=value):
                    with self.assertRaises(ValueError):RGBDBackgroundSegmenter(**{key:value})

    def test_empty_and_repeated_calls_no_state(self):
        detector=RGBDBackgroundSegmenter(fusion_mode='color_supported')
        detect(*self.gray(),detector)
        result=detect(*scene(),detector)
        self.assertFalse(result.object_mask.any())
        self.assertFalse(result.unknown_mask.any())

if __name__=='__main__':unittest.main()
