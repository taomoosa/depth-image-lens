"""Partial gripper extent diagnostics without a size-based grasp proof."""
import unittest
import numpy as np
from depth_image_lens import ObjectOnlyGraspDetector,CameraIntrinsics,TablePlane


class ObjectContextTests(unittest.TestCase):
    def scene(self):
        z=np.full((60,120),1000,np.uint16);m=np.zeros(z.shape,bool);m[15:50,40:70]=True
        hand=np.zeros_like(m);hand[5:20,10:100]=True
        diff=np.full(z.shape,100,np.uint16)
        return z,m,hand,diff
    def test_partial_gripper_pixels_cannot_inflate_width(self):
        z,m,h,d=self.scene();det=ObjectOnlyGraspDetector()
        baseline=det.detect_16uc1(z,m,difference=d)
        a=det.detect_16uc1(z,m|h,difference=d,gripper_masks={'left':h},diagnostics=True)
        self.assertEqual(a.bilateral,'unknown');self.assertEqual(a.held,'held')
        self.assertEqual(a.metrics['visible_width_px'],baseline.metrics['visible_width_px'])
        self.assertGreater(a.metrics['excluded_selected_pixels'],0)
        self.assertEqual(a.metrics['gripper_comparisons']['left']['span_x_px'],90)
    def test_absent_gripper_ratios_and_uncertain_object(self):
        z,m,h,d=self.scene();empty=np.zeros_like(m);det=ObjectOnlyGraspDetector()
        a=det.detect_16uc1(z,m,difference=d,gripper_masks={'right':empty})
        self.assertIsNone(a.metrics['gripper_comparisons']['right']['selected_component_width_ratio'])
        b=det.detect_16uc1(z,m,difference=d,uncertain_mask=m)
        self.assertEqual(b.held,'unknown');self.assertEqual(b.metrics['uncertain_selected_pixels'],int(m.sum()))
    def test_exclusion_cannot_manufacture_complete_negative(self):
        z,m,h,d=self.scene();det=ObjectOnlyGraspDetector()
        a=det.detect_16uc1(z,m|h,intrinsics=CameraIntrinsics(100,100,60,30),table=TablePlane((0,0,-1),1),
            gripper_masks={'left':h},whole_object_visible=True,mask_complete=True)
        self.assertEqual(a.held,'unknown')
    def test_wide_one_hand_counterexample_keeps_ambiguity(self):
        z,m,h,d=self.scene();m[20:50,20:100]=True
        a=ObjectOnlyGraspDetector().detect_16uc1(z,m,difference=d,gripper_masks={'left':h})
        self.assertEqual(a.bilateral,'bilateral');self.assertEqual(a.physical_grasp,'unknown')
        self.assertIn('single_hand_stretch_or_other_support_can_mimic_bilateral',a.warnings)
