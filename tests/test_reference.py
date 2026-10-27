"""Repeatable-pose reference contracts, not field segmentation accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import ReferenceGripperSegmenter,IndependentGraspDetector


def reference_scene():
    empty=np.full((80,140),2000,np.uint16);gripper=empty.copy()
    left=np.zeros(empty.shape,bool);left[20:40,20:30]=True
    right=np.zeros_like(left);right[20:40,110:120]=True
    gripper[left|right]=1000
    rgb=np.full((*empty.shape,3),60,np.uint8);rgb[left|right]=(160,120,80)
    static=np.zeros_like(left);static[60:75,5:20]=True
    rois={'left':(10,10,40,50),'right':(100,10,130,50)}
    return empty,gripper,rgb,static,rois,left,right


class ReferenceTests(unittest.TestCase):
    def prepare(self,color=True):
        empty,g,rgb,static,rois,left,right=reference_scene();d=ReferenceGripperSegmenter()
        ref=d.calibrate(empty,g,rois,static,pose_key='open',gripper_color=rgb if color else None,
            color_order='RGB',registered_to_depth=True)
        return d,ref,empty,g,rgb,left,right

    def test_reference_generated_from_coarse_boxes_and_no_object(self):
        d,ref,empty,g,rgb,left,right=self.prepare()
        np.testing.assert_array_equal(ref.hand_masks['left'],left)
        np.testing.assert_array_equal(ref.hand_masks['right'],right)
        result=d.segment(g,ref,pose_key='open',color=rgb,registered_to_depth=True)
        self.assertEqual(result.summary['sides']['left']['status'],'candidate')
        self.assertEqual(result.object_mask.sum(),0)
        json.dumps(result.to_dict(),allow_nan=False)

    def test_object_candidates_and_downstream_no_perframe_hand_masks(self):
        d,ref,empty,g,rgb,left,right=self.prepare();z=g.copy();color=rgb.copy()
        obj=np.zeros(z.shape,bool);obj[30:60,30:110]=True;z[obj]=1005;color[obj]=(20,50,190)
        result=d.segment(z,ref,pose_key='open',color=color,registered_to_depth=True,diagnostics=True)
        np.testing.assert_array_equal(result.object_mask,obj)
        grasp=IndependentGraspDetector().detect_16uc1(z,result.object_mask,result.hand_masks,
            registered_to_depth=True,observed_mask=result.observed_mask,uncertain_mask=result.uncertain_mask)
        self.assertEqual(grasp.sides['left']['physical_grasp'],'unknown')
        self.assertTrue(result.diagnostics['contact_uncertain_mask'].any())

    def test_pose_and_camera_change_abstain_without_relabeling(self):
        d,ref,empty,g,rgb,left,right=self.prepare()
        for pose,z in [('other',g),('open',g+100)]:
            a=d.segment(z,ref,pose_key=pose,color=rgb,registered_to_depth=True)
            self.assertEqual(a.summary['status'],'unknown');self.assertFalse(a.object_mask.any())
            self.assertTrue(all(v['status']=='unknown' for v in a.summary['sides'].values()))

    def test_unreported_shift_and_one_side_missing_preserve_other(self):
        d,ref,empty,g,rgb,left,right=self.prepare(False)
        for missing in (False,True):
            z=g.copy();z[left]=0 if missing else 2000
            if not missing:z[20:40,25:35]=1000
            a=d.segment(z,ref,pose_key='open',registered_to_depth=True)
            self.assertEqual(a.summary['sides']['left']['status'],'unknown')
            self.assertEqual(a.summary['sides']['right']['status'],'candidate')
            self.assertFalse(a.object_mask[10:50,10:40].any())

    def test_lighting_mismatch_and_missing_color(self):
        d,ref,empty,g,rgb,left,right=self.prepare()
        for color in (None,(rgb.astype(int)+50).clip(0,255).astype(np.uint8)):
            a=d.segment(g,ref,pose_key='open',color=color,registered_to_depth=True)
            self.assertEqual(a.summary['status'],'unknown')

    def test_same_color_occluder_requires_depth_difference(self):
        d,ref,empty,g,rgb,left,right=self.prepare()
        # Identical observable input can hide an occluder; never claim semantics.
        a=d.segment(g,ref,pose_key='open',color=rgb,registered_to_depth=True)
        self.assertIn('identical_depth_color_occlusion_is_unobservable',a.summary['warnings'])
        self.assertEqual(a.summary['physical_grasp'],'unknown')
        z=g.copy();z[left]=800
        b=d.segment(z,ref,pose_key='open',color=rgb,registered_to_depth=True)
        self.assertEqual(b.summary['sides']['left']['status'],'unknown')
        self.assertEqual(b.summary['sides']['right']['status'],'candidate')

    def test_hidden_partial_and_static_missing(self):
        d,ref,empty,g,rgb,left,right=self.prepare(False)
        a=d.segment(g,ref,pose_key='open',registered_to_depth=True,hand_visibility=('hidden','partial'))
        self.assertEqual(a.summary['sides']['left']['status'],'unknown')
        self.assertEqual(a.summary['sides']['right']['status'],'candidate')
        obs=np.ones(g.shape,bool);obs[ref.static_mask]=False
        a=d.segment(g,ref,pose_key='open',registered_to_depth=True,observed_mask=obs)
        self.assertIn('static_region_unobserved',a.summary['reasons'])

    def test_units_nonmutation_and_invalid_contracts(self):
        d,ref,empty,g,rgb,left,right=self.prepare();before=g.copy()
        a=d.segment(g,ref,pose_key='open',color=rgb,registered_to_depth=True)
        b=d.segment(g.astype(float)*.001,ref,pose_key='open',depth_scale_m=1,color=rgb[...,::-1],color_order='BGR',registered_to_depth=True)
        self.assertEqual(a.to_dict(),b.to_dict());np.testing.assert_array_equal(g,before)
        with self.assertRaises(ValueError):d.segment(g[:2],ref,pose_key='open',registered_to_depth=True)
        with self.assertRaises(ValueError):d.segment(g,ref,pose_key='open',registered_to_depth=False)
        with self.assertRaises(ValueError):d.calibrate(empty,g,{'left':(-1,0,30,40),'right':(100,10,130,50)},ref.static_mask,pose_key='x',registered_to_depth=True)

    def test_reference_arms_outside_hand_boxes_never_become_object(self):
        e,g,rgb,static,rois,left,right=reference_scene()
        arm=np.zeros(g.shape,bool);arm[:10,50:90]=True;g[arm]=1100;rgb[arm]=(130,140,150)
        d=ReferenceGripperSegmenter();ref=d.calibrate(e,g,rois,static,pose_key='A',gripper_color=rgb,registered_to_depth=True)
        a=d.segment(g,ref,pose_key='A',color=rgb,registered_to_depth=True)
        self.assertFalse(a.object_mask.any());np.testing.assert_array_equal(a.robot_mask,arm)
        z=g.copy();obj=np.zeros(g.shape,bool);obj[30:60,30:110]=True;z[obj]=1005
        color=rgb.copy();color[obj]=(20,50,190)
        b=d.segment(z,ref,pose_key='A',color=color,registered_to_depth=True)
        np.testing.assert_array_equal(b.object_mask,obj)
        self.assertFalse(b.object_mask[arm].any())

    def test_arm_mismatch_withholds_object_identity_but_keeps_hand_evidence(self):
        e,g,rgb,static,rois,left,right=reference_scene()
        arm=np.zeros(g.shape,bool);arm[:10,50:90]=True;g[arm]=1100
        d=ReferenceGripperSegmenter();ref=d.calibrate(e,g,rois,static,pose_key='A',registered_to_depth=True)
        z=g.copy();z[arm]=0;z[50:60,50:90]=1100
        a=d.segment(z,ref,pose_key='A',registered_to_depth=True)
        self.assertFalse(a.object_mask.any());self.assertTrue(a.uncertain_mask[50:60,50:90].all())
        self.assertEqual(a.summary['status'],'unknown')
        self.assertEqual(a.summary['sides']['left']['status'],'candidate')
        self.assertEqual(a.summary['sides']['right']['status'],'candidate')

    def test_missing_reference_hand_body_is_never_object(self):
        e,g,rgb,static,rois,left,right=reference_scene()
        e=e.astype(float)*.001;g=g.astype(float)*.001;current=g.copy()
        current[22:40,20:30]=.9;g[22:40,20:30]=np.nan
        d=ReferenceGripperSegmenter();ref=d.calibrate(e,g,rois,static,pose_key='A',depth_scale_m=1,registered_to_depth=True)
        a=d.segment(current,ref,pose_key='A',depth_scale_m=1,registered_to_depth=True)
        self.assertFalse(a.object_mask[left].any())
        self.assertTrue(a.uncertain_mask[22:40,20:30].all())
        self.assertEqual(a.summary['sides']['left']['reference_search_unknown_pixels'],180)
        self.assertEqual(a.summary['sides']['right']['status'],'candidate')

    def test_missing_reference_outside_search_remains_unknown(self):
        e,g,rgb,static,rois,left,right=reference_scene();g=g.astype(float)
        g[50:60,50:70]=np.nan;current=np.nan_to_num(g,nan=1000.)
        d=ReferenceGripperSegmenter();ref=d.calibrate(e,g,rois,static,pose_key='A',registered_to_depth=True)
        a=d.segment(current,ref,pose_key='A',registered_to_depth=True)
        self.assertFalse(a.object_mask[50:60,50:70].any());self.assertTrue(a.uncertain_mask[50:60,50:70].all())
