"""Component sizes and bounded longest trails on generated color/depth evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,help='Optional new directory; parent must exist')
    parser.add_argument('--max-search-states',type=int,default=100000)
    parser.add_argument('--max-search-edges',type=int,default=24)
    args=parser.parse_args()
    color=np.full((180,280,3),200,np.uint8)
    mask=np.zeros(color.shape[:2],bool);mask[8:-8,8:-8]=True
    # L, T, and a loop with two tails in disconnected regions.
    color[25:70,25:28]=30;color[67:70,25:85]=30
    color[105:108,25:90]=30;color[105:155,56:59]=30
    color[40:43,155:230]=30;color[117:120,155:230]=30
    color[40:120,155:158]=30;color[40:120,227:230]=30
    color[78:81,125:158]=30;color[78:81,227:255]=30
    depth=np.full(mask.shape,1500,np.uint16);depth[45:125,125:245]=1470
    result=OverheadBoundaryDetector(trace_boundaries=True,max_gap_px=0).detect(
        color,depth,color_order='RGB',registered_to_depth=True,depth_scale_m=.001,
        object_mask=mask,diagnostics=True)
    reports={kind:result.measure_components(kind,max_search_states=args.max_search_states,
                                           max_search_edges=args.max_search_edges)
             for kind in ('color','depth')}
    summary={kind:[{key:c[key] for key in ('id','span_xy_px','bbox_size_px','path_kind','status',
                                          'max_trail_length_px','lower_bound_length_px','upper_bound_length_px')}
                   for c in report['components']] for kind,report in reports.items()}
    print(json.dumps(summary,indent=2,allow_nan=False))
    if args.output_dir:
        import cv2
        args.output_dir.mkdir()
        result.save_debug(args.output_dir/'topology.png',color,topology_labels=True)
        (args.output_dir/'measurements.json').write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
        (args.output_dir/'topology.json').write_text(json.dumps(result.topology,indent=2,allow_nan=False)+'\n')
        for kind,report in reports.items():
            canvas=np.pad(color,((35,0),(0,0),(0,0)),constant_values=24).copy()
            cv2.putText(canvas,kind+' trails (px)',
                        (5,13),cv2.FONT_HERSHEY_SIMPLEX,.35,(235,235,235),1)
            cv2.putText(canvas,'green: maximum / orange: lower bound',
                        (5,28),cv2.FONT_HERSHEY_SIMPLEX,.32,(235,235,235),1)
            for component in report['components']:
                ink=(30,220,80) if component['exact'] else (255,150,20)
                x0,y0,x1,y1=component['bbox_xyxy']
                cv2.rectangle(canvas,(x0,y0+35),(x1,y1+35),(130,130,130),1)
                value=component['max_trail_length_px'] if component['exact'] else component['lower_bound_length_px']
                label=f"{component['id']}: {value:.1f}px" if value is not None else f"{component['id']}: no trail"
                cv2.putText(canvas,label,(x0,max(45,y0+31)),cv2.FONT_HERSHEY_SIMPLEX,.35,ink,1)
                for part in component['trail']['parts']:
                    xy=np.asarray(part['points_xy'],np.int32)+[0,35]
                    cv2.polylines(canvas,[xy],False,ink,1)
                    for point,evidence in zip(xy,part['evidence']):
                        if evidence=='inferred':canvas[point[1],point[0]]=(230,60,180)
            if not cv2.imwrite(str(args.output_dir/(kind+'-trails.png')),canvas[...,::-1]):
                raise OSError('could not save trail view')


if __name__=='__main__':main()
