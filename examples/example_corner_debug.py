"""Render synthetic Corner evidence; write only when --output is supplied."""
import argparse
import numpy as np
from depth_image_lens import OverheadCornerDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',help='New PNG path in an existing directory')
    args=parser.parse_args()
    mask=np.zeros((180,220),bool);mask[40:140,50:170]=True
    color=np.full((180,220,3),70,np.uint8);color[mask]=(30,80,200)
    depth=np.where(mask,980,1000).astype(np.uint16)
    result=OverheadCornerDetector.recommended_for_tuning().detect(
        color,depth,color_order='RGB',registered_to_depth=True,
        object_mask=mask,depth_scale_m=.001,diagnostics=True)
    overlay=result.render_debug(color,color_order='RGB',max_labels=8)
    print('RGB debug image shape:',overlay.shape)
    if args.output:
        print('Saved:',result.save_debug(args.output,color,color_order='RGB',max_labels=8))


if __name__=='__main__':main()
