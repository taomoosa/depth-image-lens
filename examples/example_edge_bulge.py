"""Synthetic edge relief, inward reference and diagnostic files; no real accuracy claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from depth_image_lens import OuterEdgeBulgeDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='New directory; parent must exist')
    args=parser.parse_args()
    y,x=np.indices((144,192))
    mask=np.zeros(y.shape,bool);mask[12:132,12:180]=True
    depth=1.-.025*np.exp(-((y-14)/8)**2)
    # Isolated false-depth spike: excluded by neighbourhood support.
    depth[18,96]=.7
    # Small missing section remains unknown, without preventing other detections.
    depth[17:21,45:51]=0
    color=np.full((*y.shape,3),50,np.uint8);color[mask]=180
    result=OuterEdgeBulgeDetector().detect(depth,mask,depth_scale_m=1,
        registered_to_mask=True,diagnostics=True)
    print(json.dumps(result.to_dict(),indent=2,allow_nan=False))
    if args.output_dir:
        import cv2
        args.output_dir.mkdir()
        result.save_debug(args.output_dir/'debug.png',color)
        (args.output_dir/'result.json').write_text(json.dumps(result.to_dict(),indent=2,allow_nan=False)+'\n')
        np.savez_compressed(args.output_dir/'arrays.npz',depth_m=depth,object_mask=mask,
            candidate_mask=result.candidate_mask,unknown_mask=result.unknown_mask,
            relative_height_m=result.relative_height_m,reference_mask=result.reference_mask,
            evaluation_mask=result.evaluation_mask,excluded_mask=result.excluded_mask,
            source_region_ids=result.diagnostics['source_region_ids'])
        if result.candidates:
            region_id=result.candidates[0]['region_ids'][0]
            local_reference=result.reference_for_region(region_id)
            overlay=color.copy();overlay[local_reference]=(40,190,250)
            overlay[result.diagnostics['source_region_ids']==region_id]=(240,180,40)
            for candidate in result.candidates:
                cv2.circle(overlay,tuple(candidate['point_xy']),2,(250,40,200),-1)
            if not cv2.imwrite(str(args.output_dir/'selected-fit-reference.png'),overlay[...,::-1]):
                raise OSError('could not save selected reference')
            np.save(args.output_dir/'selected-fit-reference.npy',local_reference)
            (args.output_dir/'selected-fit.json').write_text(json.dumps({'region_id':region_id,
                'cyan':'supported inward reference','orange':'evaluated region','magenta':'candidate representative points'})+'\n')


if __name__=='__main__':main()
