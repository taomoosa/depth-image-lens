"""Repeatable-pose reference workflow with independent synthetic captures."""
import json
import numpy as np
from depth_image_lens import ReferenceGripperSegmenter, IndependentGraspDetector


def make_example():
    empty=np.full((96,160),2000,np.uint16)
    reference_depth=empty.copy()
    reference_depth[20:45,20:32]=1000
    reference_depth[20:45,128:140]=1000
    static=np.zeros(empty.shape,bool);static[75:90,5:25]=True
    rois={'left':(10,10,42,55),'right':(118,10,150,55)}
    segmenter=ReferenceGripperSegmenter()
    reference=segmenter.calibrate(empty,reference_depth,rois,static,
        pose_key='inspection-A',registered_to_depth=True)
    current=reference_depth.copy();current[32:70,32:128]=1005
    segmentation=segmenter.segment(current,reference,pose_key='inspection-A',
        registered_to_depth=True,diagnostics=True)
    grasp=IndependentGraspDetector().detect_16uc1(current,segmentation.object_mask,
        segmentation.hand_masks,registered_to_depth=True,
        observed_mask=segmentation.observed_mask,uncertain_mask=segmentation.uncertain_mask,
        diagnostics=True)
    return current,reference,segmentation,grasp


def main():
    _,reference,segmentation,grasp=make_example()
    print(json.dumps({'reference':reference.to_dict(),'segmentation':segmentation.to_dict(),
        'grasp':grasp.to_dict()},indent=2,allow_nan=False))


if __name__=='__main__':main()
