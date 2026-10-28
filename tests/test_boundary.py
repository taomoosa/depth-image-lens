"""Analytic contracts for independent color/depth evidence, not field accuracy."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from depth_image_lens import OverheadBoundaryDetector,CameraIntrinsics


def scene():
    rgb=np.full((100,140,3),75,np.uint8);mask=np.zeros((100,140),bool);mask[15:85,20:120]=True
    rgb[mask]=(30,130,200);z=np.full(mask.shape,2.);z[mask]=1.4
    return rgb,z,mask


class BoundaryTests(unittest.TestCase):
    def run_detector(self,rgb,z,**kwargs):
        return OverheadBoundaryDetector().detect(rgb,z,color_order='RGB',registered_to_depth=True,depth_scale_m=1,**kwargs)

    def test_border_color_segmentation_flat(self):
        rgb,z,m=scene();r=self.run_detector(rgb,z)
        np.testing.assert_array_equal(r.object_mask,m)
        self.assertFalse(r.depth_boundary_mask.any());self.assertFalse(r.island_mask.any())
        self.assertEqual(r.layer_status,'unknown')

    def test_registered_background_and_table_seed(self):
        rgb,z,m=scene();bg=np.full_like(rgb,75);a=self.run_detector(rgb,z,background_color=bg)
        seed=np.zeros(m.shape,bool);seed[:10]=True;b=self.run_detector(rgb,z,table_mask=seed)
        np.testing.assert_array_equal(a.object_mask,m);np.testing.assert_array_equal(b.object_mask,m)

    def test_color_only_and_missing_depth_independent(self):
        rgb,z,m=scene();rgb[25:75,69:72]=10
        a=OverheadBoundaryDetector().detect(rgb,color_order='RGB')
        b=self.run_detector(rgb,np.full(z.shape,np.nan))
        self.assertTrue(a.color_boundary_mask.any());np.testing.assert_array_equal(a.color_boundary_mask,b.color_boundary_mask)
        self.assertEqual(b.stages['depth']['status'],'insufficient')

    def test_print_and_shadow_not_depth_confirmation(self):
        rgb,z,m=scene();rgb[25:75,69:72]=10;r=self.run_detector(rgb,z)
        self.assertTrue(r.color_boundary_mask.any());self.assertFalse(r.agreement_mask.any());self.assertFalse(r.island_mask.any())
        self.assertGreater(r.lines[0]['darkness_p95'],20)

    def test_depth_step_color_association(self):
        rgb,z,m=scene();rgb[25:75,69:72]=10;z[m&(np.indices(m.shape)[1]>=70)]-=.035
        r=self.run_detector(rgb,z)
        self.assertTrue(r.depth_boundary_mask.any());self.assertTrue(r.agreement_mask.any())

    def test_island_signed_and_contained(self):
        rgb,z,m=scene();z[38:60,56:79]-=.03;r=self.run_detector(rgb,z)
        self.assertTrue(r.island_mask[45,65]);self.assertTrue(any(x['sign']=='toward_camera' and x['contained_observed'] for x in r.islands))
        z[38:60,56:79]+=.06;r=self.run_detector(rgb,z)
        self.assertTrue(any(x['sign']=='away_from_camera' for x in r.islands))

    def test_perspective_slope_not_boundary(self):
        rgb,z,m=scene();y,x=np.indices(z.shape);z=1/(.0015*x+.002*y+.4)
        for k in [None,CameraIntrinsics(150,150,70,50)]:
            r=self.run_detector(rgb,z,intrinsics=k);self.assertFalse(r.depth_boundary_mask.any());self.assertFalse(r.island_mask.any())

    def test_holes_not_boundaries(self):
        rgb,z,m=scene();z[35:65,55:85]=0;r=self.run_detector(rgb,z)
        self.assertFalse(r.depth_boundary_mask.any());self.assertFalse(r.island_mask.any())

    def test_missing_ring_not_closed_island(self):
        rgb,z,m=scene();z[38:60,56:79]-=.03;z[37:39,56:79]=0;r=self.run_detector(rgb,z)
        self.assertTrue(any(not x['contained_observed'] for x in r.islands));self.assertFalse(r.island_mask.any())

    def test_unrelated_invalid_component_continues(self):
        rgb,z,m=scene();rgb[:,80:]=75;m[:,80:]=False;rgb[20:60,100:130]=(30,130,200);z[:,80:]=np.nan
        z[38:60,40:60]-=.03;r=self.run_detector(rgb,z)
        self.assertTrue(r.island_mask.any());self.assertTrue(any(c['status']=='insufficient' for c in r.stages['depth']['components']))

    def test_color_ambiguous_depth_explicit_domain(self):
        rgb,z,m=scene();rgb[:]=75;z[38:60,56:79]-=.03
        r=self.run_detector(rgb,z,depth_roi=m)
        self.assertFalse(r.object_mask.any());self.assertTrue(r.island_mask.any());self.assertEqual(r.stages['depth']['domain'],'caller_depth_roi')

    def test_crop_island_not_enclosed(self):
        rgb,z,m=scene();z[30:65,20:45]-=.04;r=self.run_detector(rgb,z)
        self.assertTrue(any(x['status']=='unknown' for x in r.islands));self.assertFalse(r.island_mask.any())

    def test_bgr_units_noncontiguous_nonmutation(self):
        rgb,z,m=scene();z[38:60,56:79]-=.03;original=z.copy();original_rgb=rgb.copy()
        a=self.run_detector(rgb,z);b=OverheadBoundaryDetector().detect(rgb[...,::-1],np.rint(z*1000).astype(np.uint16),color_order='BGR',registered_to_depth=True)
        np.testing.assert_array_equal(a.island_mask,b.island_mask);np.testing.assert_array_equal(rgb,original_rgb);np.testing.assert_array_equal(z,original)
        c=self.run_detector(rgb[:,::2],z[:,::2]);self.assertTrue(c.island_mask.any())

    def test_supplied_override_empty_table_samples(self):
        rgb,z,m=scene();rgb[:]=75;a=self.run_detector(rgb,z,object_mask=m)
        self.assertTrue(a.object_mask.any());b=self.run_detector(rgb,z,table_mask=np.zeros(m.shape,bool));self.assertEqual(b.stages['segmentation']['status'],'insufficient')

    def test_debug_save_no_overwrite_and_nonmutation(self):
        rgb,z,m=scene();z[38:60,56:79]-=.03;r=self.run_detector(rgb,z);old=r.island_mask.copy()
        a=r.render_debug(rgb);b=r.render_debug(rgb[...,::-1],color_order='BGR')
        np.testing.assert_array_equal(a,b[...,::-1]);np.testing.assert_array_equal(old,r.island_mask)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'debug.png';r.save_debug(p,rgb);self.assertTrue(p.stat().st_size>0)
            with self.assertRaises(FileExistsError):r.save_debug(p,rgb)

    def test_coarse_rectangle_segmentation_and_guards(self):
        rgb,z,m=scene();rgb[25:75,69:72]=10
        cv=__import__('cv2');cv.setRNGSeed(0)
        r=self.run_detector(rgb,z,object_roi=(15,10,125,90))
        self.assertGreater((r.object_mask&m).sum()/m.sum(),.95)
        self.assertTrue(r.color_boundary_mask.any())
        self.assertEqual(r.stages['segmentation']['mode'],'rectangle_grabcut')
        with self.assertRaises(ValueError):self.run_detector(rgb,z,object_roi=(0,0,140,100))
        with self.assertRaises(ValueError):self.run_detector(rgb,z,object_roi=(15,10,125,90),object_mask=m)

    def test_bad_depth_contract_preserves_color(self):
        rgb,z,m=scene();rgb[30:70,69:72]=10;d=OverheadBoundaryDetector()
        for zz,kwargs in [(z[:-1],{}),(z,{'depth_scale_m':float('nan')}),(z,{'observed_mask':np.zeros((2,3),bool)}),(np.full(z.shape,'bad'),{})]:
            scale=kwargs.pop('depth_scale_m',1.)
            r=d.detect(rgb,zz,color_order='RGB',registered_to_depth=True,depth_scale_m=scale,**kwargs)
            self.assertTrue(r.color_boundary_mask.any())
            self.assertEqual(r.stages['depth']['reason'],'invalid_depth_contract')
            self.assertFalse(r.depth_boundary_mask.any())
            __import__('json').dumps(r.to_dict(),allow_nan=False)
        with self.assertRaises(ValueError):d.detect(rgb.astype(float),z,color_order='RGB')

    def test_explicit_empty_or_filtered_mask_never_expands(self):
        rgb,z,m=scene();z[:,70:]-=.05
        for size in (0,1):
            chosen=np.zeros(m.shape,bool);chosen[45:45+size,55:55+size]=True
            r=self.run_detector(rgb,z,object_mask=chosen)
            self.assertFalse(r.depth_domain_mask.any())
            self.assertFalse(r.depth_boundary_mask.any())
            self.assertFalse(r.stages['depth']['unscoped_evidence'])
        r=self.run_detector(rgb,z,depth_roi=np.zeros(m.shape,bool))
        self.assertFalse(r.depth_domain_mask.any())

    def test_unused_scale_without_depth_does_not_abort_color(self):
        rgb,z,m=scene();rgb[30:70,69:72]=10
        r=OverheadBoundaryDetector().detect(rgb,color_order='RGB',depth_scale_m=0)
        self.assertTrue(r.color_boundary_mask.any())
        self.assertEqual(r.stages['depth']['reason'],'depth_not_supplied')

    def test_auto_ambiguous_unscoped_evidence_explicit(self):
        rgb,z,m=scene();rgb[:]=100
        r=self.run_detector(rgb,z)
        self.assertTrue(r.stages['depth']['unscoped_evidence'])
        self.assertFalse(r.stages['depth']['object_identity_available'])

    def test_configuration_and_input_guards(self):
        rgb,z,m=scene();d=OverheadBoundaryDetector.from_config({'color_distance':25})
        self.assertEqual(d.parameters['color_distance'],25)
        self.assertEqual(d.detect(rgb,z,color_order='RGB').stages['depth']['reason'],'invalid_depth_contract')
        self.assertEqual(self.run_detector(rgb,z[:-1]).stages['depth']['reason'],'invalid_depth_contract')
        with self.assertRaises(ValueError):self.run_detector(rgb,z,object_mask=m.astype(np.uint8))
        with self.assertRaises(ValueError):OverheadBoundaryDetector(color_window_px=4)
        with self.assertRaises(ValueError):OverheadBoundaryDetector(depth_step_m=np.nan)


if __name__=='__main__':unittest.main()
