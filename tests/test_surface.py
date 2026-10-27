"""Surface evidence contracts and single-layer counterexamples."""
import json
import unittest
import numpy as np
from depth_image_lens import DepthSurfaceDetector,CameraIntrinsics,TablePlane


class SurfaceTests(unittest.TestCase):
    def scene(self):
        z=np.full((72,96),2.);m=np.zeros(z.shape,bool);m[5:67,5:91]=True
        return z,m,CameraIntrinsics(150,150,48,36),TablePlane((0,0,-1),2.)
    def detect(self,z,m,k,t,**kw):return DepthSurfaceDetector().detect(z,m,depth_scale_m=1,intrinsics=k,table=t,**kw)

    def test_flat_surface_and_no_reference_are_separate(self):
        z,m,k,t=self.scene();a=self.detect(z,m,k,t)
        self.assertEqual(a.state,'no_observed_anomaly');self.assertEqual(a.layer_status,'unknown')
        b=DepthSurfaceDetector().detect(z,m,depth_scale_m=1)
        self.assertEqual(b.elevation_state,'unknown');self.assertEqual(b.shape_state,'no_observed_nonflat')
        self.assertEqual(b.state,'unknown')

    def test_sloping_calibrated_plane_is_not_nonflat(self):
        z,m,k,t=self.scene();n=np.array([.2,.1,-1.]);n/=np.linalg.norm(n);t=TablePlane(tuple(n),2.)
        y,x=np.indices(z.shape);rays=np.stack(((x-k.cx)/k.fx,(y-k.cy)/k.fy,np.ones_like(x)),axis=-1)
        z=-2/(rays@n);a=self.detect(z,m,k,t)
        self.assertEqual(a.shape_state,'no_observed_nonflat');self.assertEqual(a.elevation_state,'no_observed_elevation')

    def test_step_wrinkle_and_stack_report_evidence_not_layers(self):
        for mode in ('step','wrinkle','stack'):
            z,m,k,t=self.scene();y,x=np.indices(z.shape)
            if mode=='step':z[:,35:]=1.93
            if mode=='wrinkle':z-=.04*np.sin(x/5)
            if mode=='stack':z[m]=1.94
            a=self.detect(z,m,k,t)
            self.assertEqual(a.state,'anomaly_candidate');self.assertEqual(a.layer_status,'unknown')
            if mode!='stack':self.assertEqual(a.shape_state,'observed_nonflat')
            else:self.assertEqual(a.shape_state,'no_observed_nonflat')
            json.dumps(a.to_dict(),allow_nan=False)

    def test_single_sheet_airgap_is_indistinguishable_from_stack(self):
        z,m,k,t=self.scene();z[m]=1.94
        a=self.detect(z,m,k,t);b=self.detect(z.copy(),m,k,t)
        self.assertEqual(a.to_dict(),b.to_dict());self.assertEqual(a.layer_status,'unknown')

    def test_missing_cells_do_not_abort_other_evidence(self):
        z,m,k,t=self.scene();z[:,35:]=1.93;z[:24,:24]=np.nan
        a=self.detect(z,m,k,t,diagnostics=True)
        self.assertEqual(a.state,'anomaly_candidate')
        self.assertTrue(a.unknown_mask.any());self.assertTrue(any(c['shape_status']=='unknown' for c in a.regions))
        z[:]=0;a=self.detect(z,m,k,t);self.assertEqual(a.state,'unknown')

    def test_disjoint_surfaces_not_mixed_in_local_fit(self):
        z,m,k,t=self.scene();m[:]=False;m[2:15,2:9]=True;m[2:15,12:20]=True
        z[2:15,12:20]=1.8
        a=DepthSurfaceDetector(cell_px=24,boundary_guard_px=0).detect(z,m,depth_scale_m=1,intrinsics=k)
        self.assertEqual(a.shape_state,'no_observed_nonflat')
        self.assertEqual(len({r['component_id'] for r in a.regions}),2)

    def test_reference_units_invalid_and_crop(self):
        z,m,k,t=self.scene();z[m]=1.94
        d=DepthSurfaceDetector();a=d.detect(z,m,depth_scale_m=1,reference_depth=np.full(z.shape,2.))
        b=d.detect((z*1000).astype(np.uint16),m,reference_depth=np.full(z.shape,2000,np.uint16))
        self.assertEqual(a.to_dict(),b.to_dict())
        self.assertEqual(a.metrics['reference'],'reference_camera_z_separation_m')
        bad=np.zeros_like(m);c=d.detect(z,m,depth_scale_m=1,reference_depth=np.full(z.shape,2.),reference_valid=bad)
        self.assertEqual(c.elevation_state,'unknown')
        with self.assertRaises(ValueError):d.detect(z,m,table=t)
        with self.assertRaises(ValueError):d.detect(z,m,depth_scale_m=1,intrinsics=k,table=t,reference_depth=z)

    def test_isolated_high_noise_does_not_pool_elevation(self):
        z,m,k,t=self.scene();m[:]=False
        for y in (5,15,25,35,45):
            for x in (5,15,25,35,45):m[y,x]=True;z[y,x]=1.9
        a=self.detect(z,m,k,t);self.assertNotEqual(a.elevation_state,'observed_elevation')

    def test_strong_perspective_plane_step_detrending(self):
        h,w=96,128;k=CameraIntrinsics(160,160,63.5,47.5);y,x=np.indices((h,w))
        rays=np.stack(((x-k.cx)/k.fx,(y-k.cy)/k.fy,np.ones((h,w))),axis=-1)
        for degrees in (60,65):
            normal=np.array([np.sin(np.deg2rad(degrees)),0,-np.cos(np.deg2rad(degrees))])
            plane=TablePlane(tuple(normal),np.cos(np.deg2rad(degrees)))
            z=-plane.offset_m/(rays@normal);mask=np.ones((h,w),bool)
            a=self.detect(z,mask,k,plane)
            self.assertEqual(a.state,'no_observed_anomaly')
            self.assertFalse(a.nonflat_mask.any())
            self.assertTrue(all((cell['step_pairs'] or 0)==0 for cell in a.regions))
            z[:,55:]-=.12
            b=self.detect(z,mask,k,plane)
            self.assertEqual(b.shape_state,'observed_nonflat')
            self.assertTrue(b.nonflat_mask.any())
