"""Small synthetic SVD fit regressions, not calibration accuracy claims."""
import json
import unittest
import numpy as np
from depth_image_lens import fit_table_plane,CameraIntrinsics,TablePlane,LiftDetector
from tests.test_lift import scene


class PlaneFitTests(unittest.TestCase):
    def test_flat_and_oblique_plane(self):
        for angle in (20,60):
            z,k,expected,roi=scene(height=0,angle=angle)
            fit=fit_table_plane(z,k,roi,depth_scale_m=1)
            np.testing.assert_allclose(fit.plane.normal,expected.normal,atol=1e-12)
            self.assertAlmostEqual(fit.plane.offset_m,expected.offset_m)
            self.assertLess(fit.residual_p95_m,1e-12)
            current,_,_,_=scene(angle=angle)
            self.assertEqual(LiftDetector().detect(current,k,fit.plane,roi).state,'lifted')
            json.dumps(fit.to_dict(),allow_nan=False)
        k=CameraIntrinsics(100,100,19.5,14.5)
        fit=fit_table_plane(np.ones((30,40)),k,np.ones((30,40),bool),depth_scale_m=1)
        np.testing.assert_allclose(fit.plane.normal,(0,0,-1),atol=1e-12)
        self.assertAlmostEqual(fit.plane.offset_m,1)

    def test_noise_residual_and_warning(self):
        rng=np.random.default_rng(31);z=1+rng.normal(0,.003,(40,50))
        k=CameraIntrinsics(100,100,24.5,19.5)
        r=fit_table_plane(z,k,np.ones(z.shape,bool),depth_scale_m=1,residual_warning_m=.001)
        self.assertGreater(r.residual_p95_m,.003)
        self.assertLess(r.residual_p95_m,.01)
        self.assertGreater(r.residual_rmse_m,r.residual_median_m)
        self.assertEqual(r.warnings,('large_plane_residual',))

    def test_invalid_excluded_or_insufficient(self):
        z=np.ones((30,40));k=CameraIntrinsics(100,100,19.5,14.5);roi=np.ones(z.shape,bool)
        z[0,:]=0;z[1,:]=np.nan;z[2,:]=np.inf;z[3,:]=-1
        r=fit_table_plane(z,k,roi,depth_scale_m=1)
        self.assertEqual(r.point_count,26*40);self.assertAlmostEqual(r.valid_fraction,26/30)
        for bad in (np.zeros_like(z),np.full_like(z,np.nan)):
            with self.assertRaises(ValueError):fit_table_plane(bad,k,roi,depth_scale_m=1)
        with self.assertRaises(ValueError):fit_table_plane(z,k,np.zeros_like(roi),depth_scale_m=1)

    def test_count_scale_equivalence_and_no_mutation(self):
        k=CameraIntrinsics(100,100,19.5,14.5)
        backing=np.full((30,80),1000,np.uint16);view=backing[:,::2];mask=np.ones(view.shape,bool)
        a=fit_table_plane(view,k,mask)
        b=fit_table_plane(view.astype(float)*.001,k,mask,depth_scale_m=1)
        c=fit_table_plane(view//2,k,mask,depth_scale_m=.002)
        self.assertEqual(a.to_dict(),b.to_dict());self.assertEqual(a.to_dict(),c.to_dict())
        np.testing.assert_array_equal(backing,1000)

    def test_crop_intrinsics(self):
        z,k,_,roi=scene(height=0)
        a=fit_table_plane(z,k,roi,depth_scale_m=1)
        b=fit_table_plane(z[20:,7:],k.cropped(7,20),roi[20:,7:],depth_scale_m=1)
        np.testing.assert_allclose(a.plane.normal,b.plane.normal,atol=1e-12)
        self.assertAlmostEqual(a.plane.offset_m,b.plane.offset_m)

    def test_collinear_and_invalid_arguments(self):
        z=np.ones((30,40));k=CameraIntrinsics(100,100,19.5,14.5)
        line=np.zeros(z.shape,bool);line[15,:]=True
        with self.assertRaises(ValueError):fit_table_plane(z,k,line,depth_scale_m=1)
        roi=np.ones(z.shape,bool)
        for options in ({'depth_scale_m':0},{'min_points':True},{'min_spread_m':0},{'residual_warning_m':np.nan}):
            with self.assertRaises(ValueError):fit_table_plane(z,k,roi,**options)
        with self.assertRaises(ValueError):fit_table_plane(z,k,roi.astype(int))
        with self.assertRaises(ValueError):fit_table_plane(z.astype(complex),k,roi)

    def test_orientation(self):
        z=np.ones((30,40));k=CameraIntrinsics(100,100,19.5,14.5);roi=np.ones(z.shape,bool)
        a=fit_table_plane(z,k,roi,depth_scale_m=1)
        b=fit_table_plane(z,k,roi,depth_scale_m=1,positive_direction=(0,0,1))
        self.assertGreater(a.plane.height([[0,0,0]])[0],0)
        np.testing.assert_allclose(a.plane.normal,-np.asarray(b.plane.normal),atol=1e-12)
        self.assertAlmostEqual(a.plane.offset_m,-b.plane.offset_m)
        self.assertEqual(b.orientation,'caller_direction')
        for direction in ((0,0,0),(1,0,0),(0,np.nan,1)):
            with self.assertRaises(ValueError):fit_table_plane(z,k,roi,depth_scale_m=1,positive_direction=direction)


if __name__=='__main__':unittest.main()
