"""Minimal in-memory boundary example; generated scene is not real accuracy evidence."""
import argparse
import json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional new PNG path (parent must exist)')
    args=parser.parse_args()
    color=np.full((120,160,3),80,np.uint8)
    color[15:105,20:140]=(40,140,210)
    color[35:85,77:80]=15
    depth=np.full((120,160),1800,np.uint16)
    depth[15:105,20:140]=1400
    depth[40:80,80:110]=1370
    detector=OverheadBoundaryDetector()
    result=detector.detect(color,depth,color_order='RGB',registered_to_depth=True)
    print(json.dumps(result.to_dict(),allow_nan=False))
    if args.output:result.save_debug(args.output,color)


if __name__=='__main__':main()
