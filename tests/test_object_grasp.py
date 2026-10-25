"""Object-only suspension-rule contracts, not physical grasp accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import (ObjectOnlyGraspDetector, IndependentGraspDetector,
    CameraIntrinsics, TablePlane)


class ObjectOnlyTests(unittest.TestCase):
    def scene(self, width=50, lift=.1):
        z=np.full((50,100),2.); mask=np.zeros(z.shape,bool);mask[10:40,20:20+width]=True
        z[mask]=2.-lift; difference=np.zeros_like(z);difference[mask]=lift
        return z,mask,difference

    def detect(self,z,mask,difference,**kwargs):
        return ObjectOnlyGraspDetector().detect(z,mask,depth_unit='m',difference=difference,**kwargs)

    def test_elevated_narrow_and_wide_without_hands(self):
        for width,bilateral in [(10,'unknown'),(50,'bilateral')]:
            r=self.detect(*self.scene(width));self.assertEqual(r.held,'held');self.assertEqual(r.bilateral,bilateral)
            self.assertEqual(r.physical_grasp,'unknown');json.dumps(r.to_dict(),allow_nan=False)

    def test_missing_reference_or_depth(self):
        z,m,d=self.scene();det=ObjectOnlyGraspDetector()
        self.assertEqual(det.detect(z,m,depth_unit='m').reason,'missing_reference')
        z[:]=0;r=self.detect(z,m,d);self.assertEqual(r.held,'unknown');self.assertEqual(r.metrics['valid_pixels'],0)

    def test_zero_exclusion_is_not_negative_reference(self):
        z,m,d=self.scene();d[:]=0
        r=self.detect(z,m,d,mask_complete=True,whole_object_visible=True)
        self.assertEqual(r.held,'unknown');self.assertEqual(r.metrics['valid_fraction'],1.)
        d[m]=.1;v=np.zeros_like(m);r=self.detect(z,m,d,background_valid=v)
        self.assertEqual(r.held,'unknown')

    def test_table_rest_negative_requires_complete_observation(self):
        z,m,d=self.scene(lift=0);kw=dict(depth_unit='m',intrinsics=CameraIntrinsics(100,100,50,25),table=TablePlane((0,0,-1),2))
        det=ObjectOnlyGraspDetector()
        self.assertEqual(det.detect(z,m,**kw).held,'unknown')
        self.assertEqual(det.detect(z,m,**kw,mask_complete=True,whole_object_visible=True).held,'not_held')
        z[10,20]=0
        self.assertEqual(det.detect(z,m,**kw,mask_complete=True,whole_object_visible=True).held,'unknown')

    def test_tilted_plane_and_crop_intrinsics(self):
        k=CameraIntrinsics(150,150,50,25);n=np.array([0,.5,-1.]);n/=np.linalg.norm(n)
        y,x=np.indices((50,100));rays=np.stack(((x-50)/150,(y-25)/150,np.ones_like(x)),axis=-1)
        z=(.1-2.)/(rays@n);m=np.zeros(z.shape,bool);m[10:40,20:80]=True
        det=ObjectOnlyGraspDetector();kw=dict(depth_unit='m',table=TablePlane(tuple(n),2.))
        r=det.detect(z,m,intrinsics=k,**kw);self.assertEqual(r.held,'held');self.assertEqual(r.bilateral,'bilateral')
        c=det.detect(z[5:45,10:90],m[5:45,10:90],intrinsics=k.cropped(10,5),**kw)
        np.testing.assert_allclose(r.metrics['reference_quantiles_m'],c.metrics['reference_quantiles_m'])

    def test_crop_keeps_positive_lower_bound_not_negative(self):
        z,m,d=self.scene();r=self.detect(z[20:],m[20:],d[20:])
        self.assertEqual(r.bilateral,'bilateral');self.assertTrue(r.metrics['width_is_lower_bound'])
        r=self.detect(z[20:,40:60],m[20:,40:60],d[20:,40:60]);self.assertEqual(r.held,'held');self.assertEqual(r.bilateral,'unknown')

    def test_disconnected_regions_do_not_sum_width(self):
        z,m,d=self.scene();m[:]=False;m[10:40,10:30]=True;m[10:40,60:80]=True;d[m]=.1;z[m]=1.9
        r=self.detect(z,m,d);self.assertEqual(r.held,'held');self.assertEqual(r.bilateral,'unknown')
        self.assertEqual(len(r.metrics['components']),2)

    def test_hollow_bbox_and_single_row_noise_are_not_wide(self):
        z,m,d=self.scene();m[:]=False;m[10:40,20:24]=True;m[10:40,66:70]=True;m[39,20:70]=True
        r=self.detect(z,m,d);self.assertEqual(r.bilateral,'unknown');self.assertEqual(r.metrics['visible_width_px'],4)

    def test_depth_discontinuity_splits_width(self):
        z,m,d=self.scene();z[:,45:]=1.5;r=self.detect(z,m,d)
        self.assertEqual(r.held,'held');self.assertEqual(r.bilateral,'unknown');self.assertEqual(len(r.metrics['components']),2)

    def test_multiple_components_choose_supported_wide_region(self):
        z,m,d=self.scene();m[2:7,2:7]=True;d[m]=.1
        r=self.detect(z,m,d);self.assertEqual(r.bilateral,'bilateral');self.assertEqual(r.metrics['visible_width_px'],50)

    def test_uint16_mm_stride_no_mutation(self):
        z,m,d=self.scene();det=ObjectOnlyGraspDetector();raw=(z*1000).astype(np.uint16);delta=(d*1000).astype(np.uint16)
        raw=np.repeat(raw,2,axis=1)[:,::2];before=raw.copy()
        a=det.detect_16uc1(raw,m,difference=delta);b=det.detect(raw,m,depth_unit='mm',difference=delta)
        self.assertEqual(a.to_dict(),b.to_dict());np.testing.assert_array_equal(raw,before)

    def test_invalid_inputs_and_local_measurements_survive(self):
        z,m,d=self.scene();det=ObjectOnlyGraspDetector()
        for kw in [dict(depth_unit='counts'),dict(depth_unit='m',difference=d[:2]),dict(depth_unit='m',intrinsics=CameraIntrinsics(1,1,0,0))]:
            with self.assertRaises(ValueError):det.detect(z,m,**kw)
        with self.assertRaises(ValueError):ObjectOnlyGraspDetector(min_width_rows=0)
        z[10:35,20:70]=0;r=self.detect(z,m,d,diagnostics=True)
        self.assertEqual(r.held,'unknown');self.assertTrue(r.metrics['components']);self.assertIsNotNone(r.metrics['reference_quantiles_m'])

    def test_old_independent_hand_contract_remains_unknown_without_hands(self):
        z,m,d=self.scene();empty=np.zeros_like(m)
        r=IndependentGraspDetector().detect(z,m,{'left':empty,'right':empty},depth_unit='m',registered_to_depth=True)
        self.assertEqual([v['estimate'] for v in r.sides.values()],['unknown','unknown'])
        self.assertEqual(self.detect(z,m,d).held,'held')

    def test_sparse_wide_rows_are_not_sustained_width(self):
        z,m,d=self.scene();m[:]=False;m[10:40,20:24]=True
        for row in (10,20,30):m[row,20:70]=True
        r=self.detect(z,m,d);self.assertEqual(r.held,'held');self.assertEqual(r.bilateral,'unknown')

    def test_unobserved_pixels_and_noise_do_not_imply_holding(self):
        z,m,d=self.scene();observed=np.zeros_like(m)
        self.assertEqual(self.detect(z,m,d,observed_mask=observed).held,'unknown')
        d[:]=0;d[15,25]=.1
        self.assertEqual(self.detect(z,m,d).held,'unknown')


class LocalObservationRegionTests(unittest.TestCase):
    def scene(self, bad_value=0):
        z=np.ones((60,120));m=np.zeros(z.shape,bool);m[5:15,5:55]=True
        m[25:55,60:110]=True;z[25:55,60:110]=bad_value
        return z,m,np.full(z.shape,.1)

    def detect(self,z,m,d,**kw):
        return ObjectOnlyGraspDetector().detect(z,m,depth_unit='m',difference=d,diagnostics=True,**kw)

    def test_unrelated_missing_region_preserves_positive(self):
        for value in (0,np.nan,np.inf):
            r=self.detect(*self.scene(value))
            self.assertEqual((r.held,r.bilateral),('held','bilateral'))
            self.assertEqual(r.metrics['valid_fraction'],.25)
            self.assertEqual(r.metrics['visible_width_px'],50)
            self.assertEqual([x['valid_fraction'] for x in r.metrics['observation_regions']],[1.,0.])
            self.assertIn('local_positive_with_low_global_reference_coverage',r.warnings)
            self.assertEqual(r.physical_grasp,'unknown');json.dumps(r.to_dict(),allow_nan=False)

    def test_unrelated_valid_resting_region_does_not_dilute_lift(self):
        z,m,d=self.scene(1);d[25:55,60:110]=0
        r=ObjectOnlyGraspDetector(min_lift_fraction=.8).detect(z,m,depth_unit='m',difference=d)
        self.assertEqual(r.held,'held');self.assertEqual(r.metrics['elevated_fraction'],.25)
        self.assertEqual(r.metrics['components'][0]['region_lift_fraction'],1.)

    def test_connected_missing_region_keeps_local_coverage_guard(self):
        z,m,d=self.scene();m[14:26,54:61]=True;z[14:26,54:61]=0
        r=self.detect(z,m,d)
        self.assertEqual(r.held,'unknown');self.assertEqual(len(r.metrics['observation_regions']),1)
        self.assertIn('region_reference_coverage_low',r.metrics['components'][0]['failure_reasons'])
        self.assertTrue(r.diagnostics['elevated_mask'].any())

    def test_all_missing_or_subthreshold_islands_cannot_be_positive(self):
        z,m,d=self.scene();z[:]=0
        self.assertEqual(self.detect(z,m,d).held,'unknown')
        z[:]=1;m[:]=False;m[5:8,5:8]=True;m[20:23,60:63]=True
        r=self.detect(z,m,d);self.assertEqual(r.held,'unknown')
        self.assertEqual(sum(c['pixels'] for c in r.metrics['components']),18)
        self.assertTrue(all('component_area_low' in c['failure_reasons'] for c in r.metrics['components']))

    def test_regions_do_not_pool_width_or_object_identity(self):
        z,m,d=self.scene(1);m[:]=False;m[5:15,5:30]=True;m[25:35,60:85]=True
        r=self.detect(z,m,d);self.assertEqual((r.held,r.bilateral),('held','unknown'))
        self.assertEqual(r.metrics['visible_width_px'],25)
        self.assertIn('multiple_regions_identity_unverified',r.warnings)
        self.assertEqual({c['observation_region_id'] for c in r.metrics['components']},{1,2})

    def test_missing_reference_or_partial_observation_cannot_prove_negative(self):
        z,m,d=self.scene();det=ObjectOnlyGraspDetector()
        self.assertEqual(det.detect(z,m,depth_unit='m').held,'unknown')
        r=det.detect(z,m,depth_unit='m',intrinsics=CameraIntrinsics(100,100,60,30),
            table=TablePlane((0,0,-1),1),whole_object_visible=True,mask_complete=True)
        self.assertEqual(r.held,'unknown')
        observed=np.ones_like(m);observed[25:55,60:110]=False;z[:]=1
        r=self.detect(z,m,d,observed_mask=observed)
        self.assertEqual(r.held,'held');self.assertEqual(r.metrics['valid_fraction'],.25)
