"""Existing-API compositions and counterexamples, not real-image accuracy."""
import unittest
import numpy as np
from depth_image_lens import (RGBDBackgroundSegmenter, OuterContourDetector,
                              background_difference)
from .test_rgbd_background import scene, detect


class RGBDShadowOccluderTests(unittest.TestCase):
    def bounded(self, c, z, bc, bz, mode='depth_required', scale=.001, roi=None):
        # Difference is float even for uint16; invalid samples remain unknown.
        unit = 'mm' if scale == .001 else 'm'
        delta, valid = background_difference(z, bz, depth_unit=unit,
                                             background_unit=unit, invalid_values=(65535,))
        near = valid & (delta > .08 / scale)
        domain = ~near if roi is None else roi & ~near
        result = RGBDBackgroundSegmenter(
            fusion_mode=mode, missing_depth_policy='unknown').detect(
                c, z, background_color=bc, background_depth=bz,
                color_order='RGB', registered_to_depth=True, depth_scale_m=scale,
                work_roi=domain, invalid_values=(65535,), diagnostics=True)
        return result, near, valid

    def target(self):
        c, z, bc, bz = scene()
        z[10:25, 10:30] = 980
        c[10:25, 10:30] = (170, 50, 120)
        return c, z, bc, bz

    def test_dark_shadows_on_colored_tables_touching_target(self):
        for background in ((140,140,140), (180,150,100), (80,140,180), (80,160,100)):
            for ratio in (.3, .1, .02):
                with self.subTest(background=background, ratio=ratio):
                    c,z,bc,bz=self.target();bc[:]=background;c[:]=background
                    c[10:25,10:30]=(170,50,120)
                    c[10:25,30:45]=np.rint(np.array(background)*ratio).astype(np.uint8)
                    r,_,_=self.bounded(c,z,bc,bz)
                    np.testing.assert_array_equal(r.object_mask,z<bz)

    def test_color_seeded_unknown_alone_does_not_reject_colored_dark_shadow(self):
        c,z,bc,bz=self.target();bc[:]=(80,160,100);c[:]=bc
        c[10:25,10:30]=(170,50,120);c[10:25,30:45]=(8,16,10)
        color,_,_=self.bounded(c,z,bc,bz,'color_seeded')
        depth,_,_=self.bounded(c,z,bc,bz)
        self.assertTrue(color.object_mask[10:25,30:45].all())
        self.assertFalse((color.object_mask&depth.object_mask)[10:25,30:45].any())
        self.assertTrue(color.uncertain_mask[10:25,30:45].all())

    def test_near_occluder_touching_and_crossing_is_not_restored(self):
        for cut in ((slice(10,25),slice(30,40)),(slice(15,19),slice(8,34))):
            c,z,bc,bz=self.target();target=z<bz
            z[cut]=550;c[cut]=(35,175,205)
            baseline=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_seeded'))
            self.assertTrue(baseline.object_mask[cut].all())
            r,near,valid=self.bounded(c,z,bc,bz)
            np.testing.assert_array_equal(r.object_mask,target&~near)
            self.assertTrue(valid[cut].all())
            self.assertFalse(r.depth_valid_mask[cut].any())  # ROI-gated, not raw validity.
            self.assertFalse(r.outline_observed_mask[cut].any())
            outline=OuterContourDetector().detect(r.object_mask,observed_mask=r.outline_observed_mask)
            self.assertTrue(any(x['clipped'] for x in outline.contours))

    def test_all_missing_forms_and_reference_missing_are_unknown(self):
        for reference in (False,True):
            for missing in (0.,-1.,np.nan,np.inf,65535.):
                with self.subTest(reference=reference, missing=missing):
                    c,z,bc,bz=self.target();z=z.astype(float);bz=bz.astype(float)
                    (bz if reference else z)[12:18,15:22]=missing
                    r,near,valid=self.bounded(c,z,bc,bz)
                    hole=np.s_[12:18,15:22]
                    self.assertFalse(r.object_mask[hole].any())
                    self.assertFalse(r.outline_observed_mask[hole].any())
                    self.assertTrue(r.uncertain_mask[hole].all())
                    self.assertFalse(valid[hole].any());self.assertFalse(near[hole].any())
                    self.assertTrue(np.isnan(r.diagnostics['depth_difference_m'][hole]).all())

    def test_invalid_occluder_is_resurrected_by_default_color_fallback(self):
        c,z,bc,bz=self.target();z[4:9,40:48]=0;c[4:9,40:48]=(35,175,205)
        old=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_seeded'))
        new,_,_=self.bounded(c,z,bc,bz,'color_seeded')
        self.assertTrue(old.object_mask[4:9,40:48].all())
        self.assertFalse(new.object_mask[4:9,40:48].any())
        self.assertFalse(new.observed_mask[4:9,40:48].any())

    def test_upper_cut_equality_units_and_input_immutability(self):
        c,z,bc,bz=self.target();z[4,5:8]=(921,920,919)
        original=[a.copy() for a in (c,z,bc,bz)]
        r,near,_=self.bounded(c,z,bc,bz)
        self.assertEqual(near[4,5:8].tolist(),[False,False,True])
        # Use exact binary fractions for equality across metric units.
        for scale,zvalue,bgvalue in ((.001,937.5,1000.),(1.,.9375,1.)):
            zz=np.full(z.shape,zvalue);bb=np.full(z.shape,bgvalue)
            q,_,_=self.bounded(c,zz,bc,bb,scale=scale)
            self.assertTrue(q.object_mask.all())
        for a,b in zip((c,z,bc,bz),original):np.testing.assert_array_equal(a,b)

    def test_roi_is_respected_and_empty_domain_stays_empty(self):
        c,z,bc,bz=self.target();roi=np.ones(z.shape,bool);roi[:,20]=False
        r,_,_=self.bounded(c,z,bc,bz,roi=roi)
        self.assertFalse(r.object_mask[:,20].any());self.assertFalse(r.observed_mask[:,20].any())
        r,_,_=self.bounded(c,z,bc,bz,roi=np.zeros(z.shape,bool))
        self.assertEqual(r.stages['fusion']['status'],'empty_domain')

    def test_shadow_depth_noise_below_threshold_but_bias_is_ambiguous(self):
        c,z,bc,bz=self.target();c[10:25,30:45]=40
        z[10:25,30:45]=np.random.default_rng(41).integers(998,1003,(15,15))
        r,_,_=self.bounded(c,z,bc,bz)
        self.assertFalse(r.object_mask[10:25,30:45].any())
        z[10:25,30:45]=993
        r,_,_=self.bounded(c,z,bc,bz)
        self.assertTrue(r.object_mask[10:25,30:45].all())

    def test_same_color_retention_and_optional_color_intersection_loss(self):
        c,z,bc,bz=self.target();c[:]=bc
        d,_,_=self.bounded(c,z,bc,bz);s,_,_=self.bounded(c,z,bc,bz,'color_seeded')
        np.testing.assert_array_equal(d.object_mask,z<bz)
        self.assertFalse((s.object_mask&d.object_mask).any())
        self.assertFalse(s.outline_observed_mask[z<bz].any())

    def test_thin_high_and_same_band_occluder_limitations(self):
        for depth in (998,880):
            c,z,bc,bz=self.target();z[z<bz]=depth
            r,_,_=self.bounded(c,z,bc,bz)
            self.assertFalse(r.object_mask.any())
        c,z,bc,bz=self.target();z[4:9,40:48]=980;c[4:9,40:48]=(35,175,205)
        r,_,_=self.bounded(c,z,bc,bz)
        self.assertTrue(r.object_mask[4:9,40:48].all())

    def test_enclosed_missing_patch_is_not_confirmed_physical_hole(self):
        c,z,bc,bz=self.target();z[15:18,18:21]=0
        r,_,_=self.bounded(c,z,bc,bz)
        self.assertFalse(r.outline_observed_mask[15:18,18:21].any())
        outline=OuterContourDetector().detect(r.object_mask,observed_mask=r.outline_observed_mask)
        # Contour holes describe binary topology, not material absence.
        self.assertEqual(len(outline.holes),1)
        self.assertEqual(outline.holes[0]['role'],'hole_not_external_edge')


if __name__=='__main__':unittest.main()
