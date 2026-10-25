"""Independent image-side grasp hypotheses; generated masks, no camera or dataset."""
import json
import numpy as np
from depth_image_lens import IndependentGraspDetector


def main():
    object_mask=np.zeros((60,100),bool);object_mask[25:50,30:70]=True
    left=np.zeros_like(object_mask);left[20:35,20:30]=True
    right=np.zeros_like(object_mask);right[20:35,80:90]=True
    depth=np.full(object_mask.shape,2000,np.uint16)
    depth[object_mask]=1000;depth[left|right]=1010
    result=IndependentGraspDetector().detect_16uc1(
        depth,object_mask,{'left':left,'right':right},depth_scale_m=.001,
        registered_to_depth=True,hand_visibility=('partial','visible'),
        mask_complete=(False,True),diagnostics=True)
    print(json.dumps(result.to_dict(),indent=2,allow_nan=False))


if __name__=='__main__':main()
