"""Keep internal boundary paths while excluding outer-following color evidence."""
import argparse,json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--near-px',type=float,default=8.)
    parser.add_argument('--reject-fraction',type=float,default=.7)
    parser.add_argument('--include-holes',action='store_true')
    parser.add_argument('--output',help='New comparison PNG in an existing directory')
    args=parser.parse_args()
    mask=np.zeros((140,200),bool);mask[10:130,10:190]=True
    color=np.full((*mask.shape,3),205,np.uint8)
    color[15:18,25:175]=30  # Outer-following dark band.
    color[68:71,14:186]=30  # Internal band with endpoints near the exterior.
    result=OverheadBoundaryDetector(trace_boundaries=True).detect(
        color,color_order='RGB',object_mask=mask)
    selection=result.filter_outer_proximity(
        near_px=args.near_px,reject_fraction=args.reject_fraction,
        include_holes=args.include_holes)
    print(json.dumps(selection.summary,indent=2,allow_nan=False))
    for row in selection.records:
        print(row['channel'],row['segment_id'],row['status'],row['near_fraction'],row['reasons'])
    # Retained records refer to the unchanged original graph, not a rebuilt graph.
    candidates=[row['segment'] for row in selection.kept]
    print('Retained candidate IDs:',[s['id'] for s in candidates])
    selection.render_debug(color)
    if args.output:print('Saved',selection.save_debug(args.output,color,show_ids=True))


if __name__=='__main__':main()
