"""Ordered external contour and explicit pixel-length query; synthetic inputs."""
import argparse
import json
import numpy as np
from depth_image_lens import OuterContourDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',help='New PNG in an existing directory')
    args=parser.parse_args()
    mask=np.zeros((160,220),bool);mask[30:131,40:181]=True
    model=OuterContourDetector(corner_turn_deg=45,line_error_px=1.5)
    result=model.detect(mask,diagnostics=True)
    query=result.query_edge_length(140,unit='px',tolerance=2,min_support_px=20,
                                   require_corner_endpoints=True)
    print(json.dumps({'summary':result.summary,'query':query},allow_nan=False,indent=2))
    for contour in result.contours:
        print('Contour',contour['id'],'cyclic order:',contour['sequence'])
    image=result.render_debug(show_ids=True)
    assert image.dtype==np.uint8
    if args.output:print('Saved',result.save_debug(args.output))


if __name__=='__main__':main()
