"""Fixed geometric proposal regressions, not dataset accuracy tests."""
import importlib.util
import json
import unittest
import numpy as np
from depth_image_lens import OverheadCornerDetector
from .corner_cases import cases


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV dependency missing')
class CornerProposalTests(unittest.TestCase):
    def detect(self, name='flat', **options):
        item=next(c for c in cases() if c['name']==name)
        detector=OverheadCornerDetector(proposal_policy='multiscale',support_policy='adaptive')
        return detector.detect(item['color'],item['depth'],object_mask=item['mask'],
            color_order='RGB',registered_to_depth=True,**options)

    def test_single_policy_is_the_existing_default(self):
        c=next(cases());kw=dict(object_mask=c['mask'],color_order='RGB',registered_to_depth=True)
        self.assertEqual(OverheadCornerDetector().detect(c['color'],c['depth'],**kw).to_dict(),
            OverheadCornerDetector(proposal_policy='single').detect(c['color'],c['depth'],**kw).to_dict())

    def test_rectangle_is_not_counted_once_per_epsilon(self):
        for mode in ['rgb','refined','depth']:
            r=self.detect(mode=mode)
            self.assertEqual(r.summary['counts']['accepted'],4)
            self.assertEqual(len(r.candidates),4)
            self.assertTrue(all(c['evidence']['proposal_scales']==[.5,1.,2.,4.] for c in r.candidates))
            json.dumps(r.to_dict(),allow_nan=False)

    def test_rounded_and_concave_shapes_are_not_promoted_by_union(self):
        self.assertEqual(self.detect('rounded').summary['counts']['accepted'],0)
        r=self.detect('concave')
        for c in r.candidates:
            if c['evidence'].get('local_convex') is False:self.assertNotEqual(c['status'],'accepted')
        self.assertTrue(any(c['evidence'].get('local_convex') is False for c in r.candidates))

    def test_depth_anomalies_and_crop_are_still_retained(self):
        self.assertGreaterEqual(self.detect('invalid_depth').summary['counts']['unknown'],1)
        self.assertGreaterEqual(self.detect('internal_fold').summary['counts']['suspected'],2)
        self.assertGreaterEqual(self.detect('cropped').summary['counts']['unknown'],2)
        self.assertFalse(self.detect('hidden_aligned_layers').summary['single_layer_guarantee'])

    def test_fixed_rotations_scales_and_translations(self):
        import cv2
        for size,angle,center in [((80,60),-23,(100,100)),((100,75),0,(130,120)),((140,100),29,(150,140))]:
            points=cv2.boxPoints((center,size,angle));mask=np.zeros((280,320),np.uint8)
            cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1);mask=mask.astype(bool)
            r=OverheadCornerDetector(proposal_policy='multiscale',support_policy='adaptive').detect(
                np.zeros((*mask.shape,3),np.uint8),np.where(mask,980,1000).astype(np.uint16),
                object_mask=mask,color_order='RGB',registered_to_depth=True,mode='refined')
            accepted=np.array([c['xy'] for c in r.candidates if c['status']=='accepted'])
            self.assertEqual(len(accepted),4)
            self.assertTrue(np.all(np.min(np.linalg.norm(points[:,None]-accepted[None],axis=2),axis=1)<3))

    def test_components_do_not_merge_and_inputs_are_unchanged(self):
        mask=np.zeros((130,180),bool);mask[30:70,30:70]=True;mask[30:70,90:130]=True
        depth=np.where(mask,980,1000).astype(np.uint16);copy=depth.copy()
        r=OverheadCornerDetector(proposal_policy='multiscale',support_policy='adaptive').detect(
            np.zeros((*mask.shape,3),np.uint8),depth,object_mask=mask,color_order='RGB',registered_to_depth=True)
        self.assertEqual(r.summary['counts']['accepted'],8)
        self.assertEqual(len({c['evidence']['component_id'] for c in r.candidates}),2)
        np.testing.assert_array_equal(depth,copy)

    def test_measurements_survive_non_right_angle_rejection(self):
        import cv2
        mask=np.zeros((180,200),np.uint8);cv2.fillPoly(mask,[np.array([[40,140],[100,35],[160,140]],np.int32)],1)
        r=OverheadCornerDetector(proposal_policy='multiscale').detect(
            np.zeros((*mask.shape,3),np.uint8),np.where(mask,980,1000).astype(np.uint16),
            object_mask=mask.astype(bool),color_order='RGB',registered_to_depth=True)
        self.assertTrue(r.candidates)
        self.assertTrue(all(c['status']!='accepted' for c in r.candidates))
        self.assertTrue(any(c['stage_results']['depth']['status']=='computed' for c in r.candidates))

    def test_line_intersection_at_side_center_is_unmeasurable(self):
        from depth_image_lens.corner import _fit_corner
        horizontal=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        vertical=np.array([[0.,1.],[0.,2.],[0.,3.],[0.,4.]])
        self.assertIsNone(_fit_corner([horizontal,vertical]))

    def test_parameter_contract(self):
        for options in [dict(proposal_policy='hull'),dict(proposal_merge_px=0),dict(proposal_merge_px=float('nan'))]:
            with self.assertRaises(ValueError):OverheadCornerDetector(**options)

if __name__=='__main__':unittest.main()
