import importlib.util
import json
import unittest
import numpy as np
from depth_image_lens import OverheadCornerDetector, CameraIntrinsics
from .corner_cases import cases


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional corner OpenCV dependency not installed')
class CornerTests(unittest.TestCase):
    def detect(self,name='flat',**options):
        c=next(c for c in cases() if c['name']==name)
        args={'color_order':'RGB','registered_to_depth':True,'object_mask':c['mask']};args.update(options)
        return OverheadCornerDetector().detect(c['color'],c['depth'],**args)

    def test_flat_object_table_boundary_is_not_fold(self):
        r=self.detect()
        self.assertEqual(r.summary['counts']['accepted'],4)
        for c in r.candidates:
            self.assertAlmostEqual(c['angle_deg'],90.)
            self.assertAlmostEqual(c['evidence']['outside_minus_inside_z_m'],.02)
            self.assertEqual(c['single_layer_status'],'unknown')
        json.dumps(r.to_dict(),allow_nan=False)

    def test_internal_fold_is_suspected_locally(self):
        r=self.detect('internal_fold')
        self.assertEqual(r.summary['counts'],{'accepted':2,'suspected':2,'unknown':0})
        self.assertTrue(any('internal_depth_step_or_nonplanarity' in c['reasons'] for c in r.candidates))

    def test_original_missing_and_mixed_boundary_depth(self):
        self.assertEqual(self.detect('invalid_depth').summary['counts']['unknown'],1)
        self.assertEqual(self.detect('mixed_boundary').summary['counts']['unknown'],4)
        self.assertLessEqual(self.detect('registration_shift').summary['counts']['accepted'],2)

    def test_rounded_concave_and_cropped(self):
        self.assertEqual(self.detect('rounded').summary['counts']['accepted'],0)
        concave=self.detect('concave')
        self.assertTrue(any(not c['evidence']['convex'] and c['status']=='suspected' for c in concave.candidates))
        self.assertEqual(self.detect('cropped').summary['counts']['unknown'],2)

    def test_pattern_noise_and_hidden_layers(self):
        for name in ('pattern','noise','hidden_aligned_layers'):
            r=self.detect(name);self.assertEqual(r.summary['counts']['accepted'],4)
            self.assertFalse(r.summary['single_layer_guarantee'])
        self.assertEqual(self.detect().to_dict(),self.detect('hidden_aligned_layers').to_dict())

    def test_units_intrinsics_crop_and_noncontiguous_input(self):
        c=next(cases());d=OverheadCornerDetector();K=CameraIntrinsics(300,300,110,90)
        base=d.detect(c['color'],c['depth'],object_mask=c['mask'],color_order='RGB',registered_to_depth=True,intrinsics=K)
        crop=d.detect(c['color'][10:,10:],c['depth'][10:,10:]*.001,object_mask=c['mask'][10:,10:],color_order='RGB',registered_to_depth=True,intrinsics=K.cropped(10,10),depth_scale_m=1.)
        self.assertEqual(base.summary['counts'],crop.summary['counts'])
        np.testing.assert_allclose(np.array([a['xy'] for a in base.candidates])-10,[a['xy'] for a in crop.candidates])
        self.assertEqual(c['depth'][40,50],980.)

    def test_registration_color_depth_and_mask_contract(self):
        for opts in ({'registered_to_depth':False},{'color_order':'auto'},{'depth_scale_m':0},{'object_mask':None},{'mode':'all'}):
            with self.assertRaises(ValueError):self.detect(**opts)
        c=next(cases())
        with self.assertRaises(ValueError):self.detect(object_mask=c['mask'].astype('uint8'))
        with self.assertRaises(ValueError):OverheadCornerDetector(edge_window_px=3)
        r=OverheadCornerDetector().detect(c['color'][...,::-1],c['depth'].astype('uint16'),object_mask=c['mask'],color_order='BGR',registered_to_depth=True)
        self.assertEqual(r.summary['counts']['accepted'],4)

    def test_nonfinite_and_explicit_invalid_are_not_filled(self):
        import warnings
        c=next(cases()); depth=c['depth'].copy(); valid=np.ones(depth.shape,bool)
        depth[20:70,20:80]=np.inf;depth[20:70,140:210]=np.nan
        valid[110:170,20:80]=False
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            r=OverheadCornerDetector().detect(c['color'],depth,object_mask=c['mask'],
                valid_mask=valid,color_order='RGB',registered_to_depth=True)
        self.assertGreaterEqual(r.summary['counts']['unknown'],3)
        json.dumps(r.to_dict(),allow_nan=False)

    def test_separate_component_does_not_change_target_depth_evidence(self):
        mask=np.zeros((130,130),bool);mask[40:91,40:91]=True
        neighbor=np.zeros_like(mask);neighbor[23:38,23:38]=True
        z=np.full(mask.shape,1000,np.uint16);z[mask]=980;z[neighbor]=920
        color=np.zeros((*mask.shape,3),np.uint8);detector=OverheadCornerDetector()
        target=[]
        for m in (mask,mask|neighbor):
            r=detector.detect(color,z,object_mask=m,color_order='RGB',registered_to_depth=True)
            target.append(next(c for c in r.candidates if c['polygon_xy']==[40.,40.]))
        a,b=target
        self.assertEqual(a['status'],'accepted');self.assertEqual(a['status'],b['status'])
        for key in ('interior_points','depth_valid_fraction','boundary_valid_fraction',
                    'local_residual_p95_m','internal_step_pair_count','boundary_trend_error_p95_m'):
            self.assertEqual(a['evidence'][key],b['evidence'][key])
        self.assertEqual(a['evidence']['nearby_other_component_pixels'],0)
        self.assertGreater(b['evidence']['nearby_other_component_valid_depth_pixels'],0)

    def test_component_mask_preserves_original_holes(self):
        mask=np.zeros((130,130),bool);mask[40:91,40:91]=True;mask[44:58,44:58]=False
        z=np.where(mask,980,1000).astype(np.uint16);z[44:58,44:58]=0
        r=OverheadCornerDetector().detect(np.zeros((*mask.shape,3),np.uint8),z,
            object_mask=mask,color_order='RGB',registered_to_depth=True,diagnostics=True)
        self.assertFalse(r.diagnostics['component_labels'][44:58,44:58].any())
        self.assertFalse(r.diagnostics['interior_mask'][44:58,44:58].any())
        target=next(c for c in r.candidates if c['polygon_xy']==[40.,40.])
        yy,xx=np.indices(mask.shape)
        expected=r.diagnostics['interior_mask']&((xx-40)**2+(yy-40)**2<=24**2)
        self.assertEqual(target['evidence']['interior_points'],int(expected.sum()))
        self.assertEqual(target['evidence']['depth_valid_fraction'],1.)

    def test_small_corner_reports_effective_line_support(self):
        mask=np.zeros((60,60),bool);mask[20:39,20:39]=True
        r=OverheadCornerDetector().detect(np.zeros((*mask.shape,3),np.uint8),
            np.full(mask.shape,1000,np.uint16),object_mask=mask,
            color_order='RGB',registered_to_depth=True,mode='refined')
        self.assertEqual(r.summary['counts']['unknown'],4)
        for c in r.candidates:
            self.assertEqual(c['reasons'],['insufficient_local_line_support'])
            self.assertLess(min(c['evidence']['short_edge_support_counts']),4)
            self.assertAlmostEqual(c['evidence']['short_edge_window_px'],c['evidence']['edge_window_px']*.6)
        json.dumps(r.to_dict(),allow_nan=False)



if __name__=='__main__':unittest.main()
