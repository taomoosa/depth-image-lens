"""Synthetic evidence and counterexamples; no real scene or grasp accuracy claim."""
import json
import unittest
import numpy as np
from depth_image_lens import OuterEdgeBulgeDetector, CameraIntrinsics


class EdgeBulgeTests(unittest.TestCase):
    def scene(self):
        y,x=np.indices((120,160));mask=np.zeros((120,160),bool);mask[12:108,12:148]=True
        z=np.ones(mask.shape);return z,mask,y,x

    def detect(self,z,m,**kw):
        return OuterEdgeBulgeDetector().detect(z,m,depth_scale_m=1,registered_to_mask=True,**kw)

    def bump(self,z,y):
        return z-.025*np.exp(-((y-14)/8)**2)

    def test_smooth_attached_bulge_and_sign(self):
        z,m,y,x=self.scene();a=self.detect(self.bump(z,y),m,diagnostics=True)
        self.assertEqual(a.state,'candidate');self.assertGreater(a.candidate_mask.sum(),300)
        self.assertTrue(all(c['median_height_m']>.005 for c in a.candidates))
        self.assertFalse((a.candidate_mask&~m).any());self.assertFalse((a.candidate_mask&a.unknown_mask).any())
        for c in a.candidates:
            px,py=c['point_xy'];self.assertTrue(a.candidate_mask[py,px])
        json.dumps(a.to_dict(),allow_nan=False)
        self.assertTrue(a.reference_mask.any());self.assertTrue(a.render_debug(panel=False).size)

    def test_flat_tilt_dip_and_absolute_elevation_not_bulges(self):
        z,m,y,x=self.scene()
        for depth in (z,z-.1,1/(1+.001*x+.0005*y),z+.025*np.exp(-((y-14)/8)**2)):
            with self.subTest():
                a=self.detect(depth,m);self.assertFalse(a.candidate_mask.any())

    def test_intrinsics_and_units(self):
        z,m,y,x=self.scene();z=self.bump(z,y);k=CameraIntrinsics(200,200,80,60)
        a=self.detect(z,m,intrinsics=k)
        b=OuterEdgeBulgeDetector().detect(z*1000,m,depth_scale_m=.001,registered_to_mask=True,intrinsics=k)
        np.testing.assert_array_equal(a.candidate_mask,b.candidate_mask)
        self.assertTrue(a.candidate_mask.any());self.assertEqual(a.metrics['height_units'],'normal_m_toward_camera')

    def test_known_normal_height_and_no_outside_depth_dependency(self):
        z,m,y,x=self.scene();k=CameraIntrinsics(200,220,80,60)
        a,b,c=.001,.0005,1.;inv=a*x+b*y+c
        norm=np.linalg.norm([a*k.fx,b*k.fy,a*k.cx+b*k.cy+c])
        z=1/inv;raised=m & (y<26);z[raised]=(1-.012*norm)/inv[raised]
        r=self.detect(z,m,intrinsics=k)
        self.assertTrue(r.candidate_mask.any())
        np.testing.assert_allclose(r.relative_height_m[r.candidate_mask],.012,atol=1e-11)
        z[~m]=.1;s=self.detect(z,m,intrinsics=k)
        np.testing.assert_array_equal(r.candidate_mask,s.candidate_mask)

    def test_noise_floor_and_small_region_reason(self):
        z,m,y,x=self.scene();z-=.003*np.exp(-((y-14)/8)**2)
        model=OuterEdgeBulgeDetector(min_height_m=.001)
        a=model.detect(z,m,depth_scale_m=1,registered_to_mask=True)
        self.assertEqual(a.metrics['threshold_m'],.004);self.assertFalse(a.candidate_mask.any())
        z=np.ones(m.shape);z[15:19,40:44]=.98
        b=self.detect(z,m)
        self.assertFalse(b.candidate_mask.any())
        self.assertTrue(any(r['reason']=='region_too_small' for r in b.rejections))

    def test_isolated_spikes_are_not_candidates(self):
        z,m,y,x=self.scene();z[16,40]=.8;z[18,100]=.7;z[30,16]=.6
        a=self.detect(z,m);self.assertFalse(a.candidate_mask.any());self.assertTrue(a.unknown_mask[16,40])

    def test_boundary_mixed_depth_guard(self):
        z,m,y,x=self.scene();z[12:15,12:148]=.5
        a=self.detect(z,m);self.assertFalse(a.candidate_mask.any());self.assertTrue(a.excluded_mask[12:15,12:148].all())

    def test_narrow_object_has_no_inward_reference(self):
        z,m,y,x=self.scene();m[:]=False;m[12:30,12:148]=True;z=self.bump(z,y)
        a=self.detect(z,m);self.assertEqual(a.state,'unknown');self.assertFalse(a.candidate_mask.any())
        self.assertTrue(any(r['reason']=='insufficient_inward_support' for r in a.regions))

    def test_hole_relief_not_outer_edge(self):
        z,m,y,x=self.scene();m[45:75,60:100]=False
        d=np.maximum(np.maximum(60-x,x-99),np.maximum(45-y,y-74))
        z-=.025*np.exp(-(np.maximum(d,0)/4)**2)
        a=self.detect(z,m);self.assertFalse(a.candidate_mask.any())

    def test_disconnected_components_keep_references_separate(self):
        z,m,y,x=self.scene();m[:,76:85]=False;z[:,85:]=.8;z[:,:76]=self.bump(z,y)[:,:76]
        a=self.detect(z,m);self.assertTrue(a.candidate_mask[:,:76].any());self.assertFalse(a.candidate_mask[:,85:].any())

    def test_frame_cut_is_excluded_other_edge_continues(self):
        z,m,y,x=self.scene();m[:12,12:148]=True;z-=.025*np.exp(-(y/8)**2);z-=.025*np.exp(-((y-105)/8)**2)
        a=self.detect(z,m,diagnostics=True)
        self.assertFalse(a.candidate_mask[:17].any());self.assertTrue(a.candidate_mask[90:].any())
        self.assertTrue(a.diagnostics['unsafe_boundary_mask'][0].any())

    def test_unknown_observation_edge_is_excluded(self):
        z,m,y,x=self.scene();obs=np.ones(m.shape,bool);obs[:12]=False
        a=self.detect(self.bump(z,y),m,observed_mask=obs,diagnostics=True)
        self.assertFalse(a.candidate_mask[:28].any());self.assertTrue(a.excluded_mask[12:28,30:100].any())

    def test_missing_context_and_missing_candidate(self):
        z,m,y,x=self.scene();z=self.bump(z,y);valid=np.ones(m.shape,bool);valid[32:70]=False
        a=self.detect(z,m,depth_valid_mask=valid)
        self.assertTrue(a.unknown_mask[16:22,60:100].any());self.assertFalse(a.candidate_mask[16:22,60:100].any())
        valid[:]=True;valid[16:20,50:80]=False
        b=self.detect(z,m,depth_valid_mask=valid)
        self.assertFalse(b.candidate_mask[16:20,50:80].any());self.assertTrue(b.candidate_mask.any())

    def test_partial_context_and_all_invalid(self):
        z,m,y,x=self.scene();z=self.bump(z,y);valid=np.ones(m.shape,bool);valid[40,::5]=False
        a=self.detect(z,m,depth_valid_mask=valid);self.assertTrue(a.candidate_mask.any())
        for value in (0,np.nan,np.inf):
            b=self.detect(np.full(m.shape,value),m);self.assertEqual(b.state,'unknown');self.assertFalse(b.candidate_mask.any())

    def test_nonplanar_context_is_unknown(self):
        z,m,y,x=self.scene();z=self.bump(z,y);z[33:65]+=.02*np.sin(x[33:65]/3)
        a=self.detect(z,m)
        self.assertTrue(any(r['reason'] in ('inward_reference_not_planar','insufficient_inward_support') for r in a.regions))
        self.assertTrue(a.unknown_mask[16:25,40:120].any())

    def test_concave_shape_and_empty(self):
        z,m,y,x=self.scene();m[12:65,65:95]=False;z=self.bump(z,y)
        a=self.detect(z,m);self.assertFalse((a.candidate_mask&~m).any());self.assertTrue(a.candidate_mask.any())
        b=self.detect(z,np.zeros_like(m));self.assertEqual(b.state,'unknown');self.assertFalse(b.candidates)

    def test_broad_relief_and_detached_peak_not_guaranteed(self):
        z,m,y,x=self.scene();a=self.detect(z-.025*np.exp(-((y-14)/40)**2),m)
        # Broad curvature can enter the inward baseline: never claim a calibrated absolute height.
        self.assertIn('broad_bulge_can_contaminate_reference',a.metrics['warnings'])
        b=self.detect(z-.025*np.exp(-((y-25)/2)**2),m)
        self.assertFalse(b.candidate_mask.any())

    def test_missing_strip_cannot_borrow_reference_across_cell(self):
        m=np.zeros((144,180),bool);m[24:132,12:168]=True
        z=np.ones(m.shape);z[:32]=.97
        for source in ('observation','depth_valid','zero_depth'):
            obs=np.ones(m.shape,bool);valid=obs.copy();depth=z.copy()
            if source=='observation':obs[32:35]=False
            elif source=='depth_valid':valid[32:35]=False
            else:depth[32:35]=0
            a=self.detect(depth,m,observed_mask=obs,depth_valid_mask=valid)
            with self.subTest(source=source):
                self.assertFalse(a.candidate_mask[27:32,40:140].any())
                self.assertTrue(a.unknown_mask[27:32,40:140].all())

    def test_candidate_source_regions_and_local_context(self):
        z,m,y,x=self.scene();a=self.detect(self.bump(z,y),m,diagnostics=True)
        regions={r['id']:r for r in a.regions}
        for c in a.candidates:
            x0,y0,x1,y1=c['bbox_xyxy']
            ids=np.unique(a.diagnostics['source_region_ids'][y0:y1+1,x0:x1+1][a.candidate_mask[y0:y1+1,x0:x1+1]])
            self.assertEqual(c['region_ids'],[int(v) for v in ids if v>0])
            for rid in c['region_ids']:
                self.assertEqual(regions[rid]['status'],'computed')
                reference=a.reference_for_region(rid)
                self.assertTrue(reference.any());self.assertFalse((reference & ~m).any())
                self.assertTrue(np.all(a.diagnostics['outer_distance_px'][reference]>19))
                reference[:]=False;self.assertTrue(a.reference_for_region(rid).any())
        with self.assertRaises(ValueError):a.reference_for_region(-1)
        b=self.detect(self.bump(z,y),m)
        with self.assertRaises(ValueError):b.reference_for_region(b.regions[0]['id'])

    def test_inputs_unchanged_validation(self):
        z,m,y,x=self.scene();z=self.bump(z,y);zc=z.copy();mc=m.copy();a=self.detect(z,m)
        np.testing.assert_array_equal(z,zc);np.testing.assert_array_equal(m,mc)
        a.object_mask[:]=False;np.testing.assert_array_equal(m,mc)
        with self.assertRaises(ValueError):OuterEdgeBulgeDetector().detect(z,m,depth_scale_m=1)
        for kwargs in (dict(edge_guard_px=6),dict(band_px=True),dict(context_px=100),dict(min_height_m=0),dict(depth_noise_m=-1),dict(min_support_fraction=.5),dict(min_context_fraction=2),dict(local_depth_tolerance_m=float('nan'))):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):OuterEdgeBulgeDetector(**kwargs)
        with self.assertRaises(ValueError):self.detect(z,m,intrinsics={})
        with self.assertRaises(ValueError):self.detect(z,m,observed_mask=np.ones((3,3),bool))


if __name__=='__main__':unittest.main()
