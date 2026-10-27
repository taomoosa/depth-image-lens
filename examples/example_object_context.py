"""Pixel extents aid review; a wide object does not prove two physical grasps."""
import json
import numpy as np
from depth_image_lens import ObjectOnlyGraspDetector


def make_examples():
    depth=np.full((72,120),1000,np.uint16);difference=np.full(depth.shape,100,np.uint16)
    partial=np.zeros(depth.shape,bool);partial[:15,5:110]=True
    rows={}
    for name,width in [('narrow_partial_gripper',28),('wide_one_hand_counterexample',70),('no_gripper',55)]:
        obj=np.zeros(depth.shape,bool);obj[10:60,25:25+width]=True
        hands={} if name=='no_gripper' else {'left':partial}
        selected=obj if not hands else obj|partial
        result=ObjectOnlyGraspDetector().detect_16uc1(depth,selected,difference=difference,
            gripper_masks=hands,diagnostics=True)
        # A review flag, not a new classifier or calibrated probability.
        needs_review=(result.bilateral=='bilateral' or
            result.metrics.get('excluded_selected_pixels',0)>0)
        rows[name]=(depth,result,needs_review)
    return rows


def main():
    print(json.dumps({k:{'result':r.to_dict(),'manual_review':v} for k,(_,r,v) in make_examples().items()},indent=2,allow_nan=False))


if __name__=='__main__':main()
