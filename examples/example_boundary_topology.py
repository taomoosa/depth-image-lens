"""Adjustable color/depth centerline graphs on an analytic scene."""
import argparse,json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',help='Optional new debug PNG path; parent must exist')
    p.add_argument('--corner-turn-deg',type=float,default=45)
    p.add_argument('--max-gap-px',type=int,default=6)
    p.add_argument('--junction-cluster-px',type=int,default=0,help='Optional grouping of compatible nearby junctions; raw graph preserved')
    args=p.parse_args()
    color=np.full((120,180,3),200,np.uint8)
    mask=np.zeros(color.shape[:2],bool);mask[8:-8,8:-8]=True
    color[25:85,35:38]=30;color[82:85,35:140]=30
    color[82:85,82:85]=200  # known gap; optional connection remains inferred
    depth=np.full(mask.shape,1500,np.uint16);depth[:,110:]-=30
    model=OverheadBoundaryDetector(trace_boundaries=True,color_weak_ratio=.5,
                                  depth_weak_ratio=.5,max_gap_px=args.max_gap_px,
                                  corner_turn_deg=args.corner_turn_deg,junction_cluster_px=args.junction_cluster_px)
    result=model.detect(color,depth,color_order='RGB',registered_to_depth=True,
                        object_mask=mask,diagnostics=True)
    print(json.dumps(result.topology,allow_nan=False))
    # Graph IDs are local to each evidence type and current result, not physical IDs.
    for kind,graph in result.topology.items():
        print(kind,graph['counts'])
        for segment in graph['segments']:
            points_xy=np.asarray(segment['points_xy'])
            assert points_xy.shape[1]==2
    if args.output:result.save_debug(args.output,color,topology_labels=True)


if __name__=='__main__':main()
