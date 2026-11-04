"""Synthetic partial-shirt outline matching; no real accuracy claim.

Run with PYTHONPATH=src python examples/example_outline_match.py
Add --output-dir /existing/parent/new-folder to save JSON and diagnostic PNGs.

For existing Boundary/Corner/RGBD/mask/contour result classes, run
example_outline_match_sources.py; it demonstrates source-specific observation
handling and both mask-to-outline and direct closed-point-list routes.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from depth_image_lens import OuterContourDetector, OutlineArticulation


def cases():
    target = np.array([[60,155], [60,70], [30,85], [20,60], [65,30], [90,35]], float)
    shirt = np.vstack([target, [[110,35],[135,30],[180,60],[170,85],[140,70],[140,155]]])
    articulation = OutlineArticulation(2, 3, (62.5,50), max_angle_deg=30, angle_step_deg=10)
    examples = [('reference', shirt, 'matched')]
    for name, angle, expected in [('sleeve_minus25',-25,'matched'), ('sleeve_plus23',23,'matched'),
                                  ('sleeve_fold90',90,'not_matched')]:
        theta = np.radians(angle)
        rotation = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
        p = shirt.copy()
        p[2:4] = (p[2:4]-[62.5,50]) @ rotation.T + [62.5,50]
        examples.append((name,p,expected))
    theta = np.radians(-33)
    rotation = np.array([[np.cos(theta),-np.sin(theta)], [np.sin(theta),np.cos(theta)]])
    examples.append(('scale_rotate', .72*(shirt-[100,100])@rotation.T+[110,115], 'matched'))
    u = np.linspace(0,1,61)
    edge = shirt[0]*(1-u[:,None])+shirt[1]*u[:,None]
    edge[:,0] += 6*np.sin(14*np.pi*u)*np.sin(np.pi*u)**2
    examples.append(('local_folds', np.vstack([edge[:-1],shirt[1:]]), 'not_matched'))
    examples.append(('missing_sleeve', np.delete(shirt,[2,3],axis=0), 'not_matched'))
    examples.append(('rectangle', np.array([[35,35],[175,35],[175,180],[35,180]]), 'not_matched'))
    return target, articulation, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(exist_ok=False)
    target, articulation, examples = cases()
    records, tiles = [], []
    for name, polygon, expected in examples:
        mask = np.zeros((240,240),np.uint8)
        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
        outline = OuterContourDetector().detect(mask > 0)
        result = outline.match_partial(target, articulation=articulation)
        print(f'{name:20s} {result.status:12s} score={result.best["score"]:.3f} expected={expected}')
        records.append(dict(case=name, expected_synthetic=expected, result=result.to_dict()))
        color = np.repeat(np.where(mask[...,None],90,25).astype(np.uint8),3,axis=2)
        overlay = result.render_debug(color)
        cv2.putText(overlay,name,(5,232),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        tiles.append(overlay)
        if args.output_dir is not None:
            result.save_debug(args.output_dir/f'{name}.png',color)
    if args.output_dir is not None:
        grid = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
        cv2.imwrite(str(args.output_dir/'comparison.png'),grid[...,::-1])
        (args.output_dir/'results.json').write_text(json.dumps(dict(
            evidence='synthetic_contracts_not_real_accuracy', target_points_xy=target.tolist(), cases=records),
            indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
