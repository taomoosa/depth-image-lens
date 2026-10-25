"""Synthetic camera-frame height regressions; not real-scene accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import CameraIntrinsics, TablePlane, LiftDetector


def scene(height=.12, angle=20):
    k = CameraIntrinsics(100, 100, 39.5, 29.5)
    n = np.array([0, -np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))])
    plane = TablePlane(tuple(n), .8)
    rays = k.deproject(np.ones((60, 80)))
    z = (height-plane.offset_m)/(rays@n)
    mask = np.zeros(z.shape, bool); mask[22:55, 12:68] = True
    return z, k, plane, mask


class LiftTests(unittest.TestCase):
    def test_oblique_signed_height(self):
        for angle in (10, 20, 35, 60):
            z,k,p,m = scene(angle=angle)
            r = LiftDetector().detect(z,k,p,m)
            self.assertEqual(r.state, 'lifted')
            self.assertAlmostEqual(r.height_quantiles_m[1], .12)
            self.assertEqual(r.whole_object_lifted, 'unknown')
            self.assertEqual(r.validation_status, 'not_validated')
            json.dumps(r.to_dict(), allow_nan=False)

    def test_table_thickness_and_uncertain_threshold(self):
        for height in (0, .003, .005):
            r = LiftDetector().detect(*scene(height=height))
            self.assertEqual((r.state, r.whole_object_lifted), ('not_lifted', 'no'))
        self.assertEqual(LiftDetector().detect(*scene(height=.025)).state, 'unknown')

    def test_lower_two_thirds_crop_and_full_visibility(self):
        z,k,p,m = scene()
        r = LiftDetector().detect(z[20:], k.cropped(0,20), p, m[20:])
        self.assertEqual((r.state,r.whole_object_lifted), ('lifted','unknown'))
        self.assertAlmostEqual(r.height_quantiles_m[1], .12)
        cropped_k = k.cropped(7,20)
        np.testing.assert_allclose(cropped_k.deproject(z[20:,7:]), k.deproject(z)[20:,7:])
        self.assertEqual(LiftDetector().detect(z[20:,7:],cropped_k,p,m[20:,7:]).state,'lifted')
        self.assertEqual(LiftDetector().detect(z,k,p,m,whole_object_visible=True).whole_object_lifted,'yes')

    def test_missing_depth_and_visibility(self):
        z,k,p,m = scene()
        for value in (0,np.nan,np.inf,-1):
            r = LiftDetector().detect(np.full_like(z,value),k,p,m)
            self.assertEqual(r.state,'unknown');json.dumps(r.to_dict(),allow_nan=False)
        self.assertEqual(LiftDetector().detect(z,k,p,m,observed_mask=np.zeros_like(m)).state,'unknown')
        for value in (0,np.nan):
            selected_missing = z.copy();selected_missing[m] = value
            r = LiftDetector().detect(selected_missing,k,p,m)
            self.assertEqual((r.state,r.whole_object_lifted),('unknown','unknown'))
            self.assertEqual(r.valid_fraction,0.)
        z[30,30]=0
        r=LiftDetector().detect(z,k,p,m,whole_object_visible=True)
        self.assertEqual((r.state,r.whole_object_lifted),('lifted','unknown'))

    def test_excluded_robot_and_partial_contact(self):
        z,k,p,m=scene()
        self.assertEqual(LiftDetector().detect(z,k,p,m,exclude_mask=m).state,'unknown')
        flat,_,_,_=scene(height=0);z[35:]=flat[35:]
        r=LiftDetector().detect(z,k,p,m)
        self.assertEqual((r.state,r.whole_object_lifted),('lifted','no'))
        self.assertGreater(r.near_table_fraction,.3)

    def test_flying_pixel_not_lift_and_z_difference_not_height(self):
        z,k,p,m=scene(height=0);high,_,_,_=scene();z[30,30]=high[30,30]
        self.assertEqual(LiftDetector().detect(z,k,p,m).state,'not_lifted')
        empty,_,_,_=scene(height=0)
        self.assertGreater(np.median((empty-high)[m]),.2)
        self.assertAlmostEqual(np.median(p.height(k.deproject(high))[m]),.12)

    def test_uint16_metre_equivalence_and_input_not_modified(self):
        # Exact representable values avoid testing sensor quantization as conversion error.
        k=CameraIntrinsics(100,100,9.5,9.5);p=TablePlane((0,0,-1),1)
        for scale,counts in ((.001,800),(.002,400)):
            backing=np.full((20,40),counts,np.uint16);z=backing[:,::2]
            mask=np.ones(z.shape,bool)
            r=LiftDetector().detect_16uc1(z,k,p,mask,depth_scale_m=scale)
            expected=LiftDetector().detect(z.astype(float)*scale,k,p,mask)
            self.assertEqual(r.to_dict(),expected.to_dict())
            np.testing.assert_array_equal(backing,counts)
        z[0,0]=0
        self.assertEqual(LiftDetector().detect_16uc1(z,k,p,mask,whole_object_visible=True).whole_object_lifted,'unknown')

    def test_invalid_geometry_masks_and_encoding(self):
        z,k,p,m=scene()
        for n,offset in (((0,0,0),1),((0,0,np.nan),1),((0,0,-1),np.inf)):
            with self.assertRaises(ValueError):TablePlane(n,offset)
        with self.assertRaises(ValueError):CameraIntrinsics(0,1,0,0)
        with self.assertRaises(ValueError):LiftDetector().detect(z,k,p,m.astype(int))
        with self.assertRaises(ValueError):LiftDetector().detect_16uc1(z,k,p,m)
        with self.assertRaises(ValueError):LiftDetector().detect_16uc1(np.ones(z.shape,np.uint16),k,p,m,depth_scale_m=0)
        with self.assertRaises(ValueError):LiftDetector(min_pixels=True)
        self.assertAlmostEqual(TablePlane((0,0,-2),2).height([[0,0,.8]])[0],.2)
        points = k.deproject(z)[m]
        for factor in (.01,2.,100.):
            scaled = TablePlane(tuple(factor*np.asarray(p.normal)),factor*p.offset_m)
            np.testing.assert_allclose(scaled.height(points),p.height(points),atol=1e-12)
            self.assertEqual(LiftDetector().detect(z,k,scaled,m).state,'lifted')
        # A negative coefficient multiplier reverses the declared above-table side.
        flipped = TablePlane(tuple(-np.asarray(p.normal)),-p.offset_m)
        np.testing.assert_allclose(flipped.height(points),-p.height(points),atol=1e-12)

    def test_nonfinite_height_is_unknown_and_json_safe(self):
        z=np.full((10,10),1e308);k=CameraIntrinsics(1e-308,1e-308,0,0)
        r=LiftDetector().detect(z,k,TablePlane((0,0,-1),1),np.ones(z.shape,bool))
        self.assertEqual(r.reason,'nonfinite_height_geometry')
        json.dumps(r.to_dict(),allow_nan=False)


if __name__=='__main__':unittest.main()
