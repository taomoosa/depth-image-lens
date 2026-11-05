"""Compare corner/edge registration with legacy arclength matching on synthetic data.

Run: PYTHONPATH=src python examples/example_corner_outline_match.py
Optional --output-dir /existing/parent/new-directory saves full JSON and PNGs.
No real-image labels or accuracy claims. Timings include one call, not a benchmark.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cv2

from depth_image_lens import (
    CornerOutlineMatcher, PartialOutlineMatcher, OutlineArticulation, OuterContourDetector,
)


def rotation(degrees):
    theta = np.radians(degrees)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def make_cases():
    target = np.array([[60,155], [60,70], [30,85], [20,60], [65,30], [90,35]], float)
    current = np.vstack([target, [[110,35], [135,30], [180,60], [170,85], [140,70], [140,155]]])
    art = OutlineArticulation(2, 3, (62.5,50), max_angle_deg=30, angle_step_deg=10)
    examples = [('identity', current, 'matched'),
                ('uniform_scale_rotate', .8*(current-[100,100])@rotation(35).T+[120,120], 'matched'),
                ('axis_scale', current*[1.15,.95], 'matched')]
    for degrees, expected in [(20, 'matched'), (90, 'not_matched')]:
        q = current.copy()
        q[2:4] = (q[2:4]-[62.5,50])@rotation(degrees).T+[62.5,50]
        examples.append((f'sleeve_{degrees}', q, expected))
    examples.extend([
        ('excess_axis_scale', current*[1.6,.7], 'not_matched'),
        ('shear', current@np.array([[1.,.5], [0,1.]]).T, 'not_matched'),
        ('missing_corner', np.delete(current, 2, axis=0), 'not_matched'),
        ('extra_corner', np.vstack([current[:2], [[48,58], [40,75]], current[2:]]), 'not_matched'),
    ])
    u = np.linspace(0, 1, 61)
    edge = current[0]*(1-u[:,None])+current[1]*u[:,None]
    edge[:,0] += 6*np.sin(14*np.pi*u)*np.sin(np.pi*u)**2
    examples.append(('local_folds', np.vstack([edge[:-1], current[1:]]), 'not_matched'))
    return target, art, examples


def run_examples(output_dir=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=False)
    target, articulation, cases = make_cases()
    corner = CornerOutlineMatcher(
        scale_range=(.5,2.), axis_ratio_range=(.75,4/3), max_rotation_deg=180.,
        fraction_range=(.15,.65), axis_angle_deg=0.,
    )
    legacy = PartialOutlineMatcher()
    records = []
    for name, points, expected in cases:
        # Direct closed contour. Do not repeat the first point at its end.
        models = {}
        for label, matcher in [('corner', corner), ('legacy', legacy)]:
            before = time.perf_counter()
            result = matcher.match_contour(target, points, articulation=articulation)
            models[label] = dict(seconds=time.perf_counter()-before, result=result.to_dict())
            if label == 'corner':
                assert result.status == expected, (name, result.status, expected)
                if output_dir is not None:
                    color = np.full((300,360,3), 25, np.uint8)
                    cv2.fillPoly(color, [np.rint(points).astype(np.int32)], (75,75,75))
                    result.save_debug(output_dir/f'{name}.png', color)
        best = models['corner']['result']['best']
        print(f"{name:23s} corner={models['corner']['result']['status']:12s} "
              f"legacy={models['legacy']['result']['status']:12s} "
              f"score={best['score']:.3f} sx={best['scale_x']:.3f} sy={best['scale_y']:.3f} "
              f"rotation={best['rotation_deg']:.1f} sleeve={best['articulation_angle_deg']:.1f}")
        records.append(dict(case=name, evidence='synthetic_continuous_polygon', expected_corner=expected, **models))

    # Same-image dense reference, selected by the detector's corner IDs. These
    # IDs only apply to THIS generated image. Both endpoint corners are excluded
    # from the matcher's transform fit; intervening raw points preserve curves.
    mask = np.zeros((240,260), np.uint8)
    cv2.fillPoly(mask, [cases[0][1].astype(np.int32)], 1)
    outline = OuterContourDetector(corner_turn_deg=35).detect(mask > 0)
    contour = outline.largest_contour()
    corners = {k['id']: k for k in contour['corners']}
    points = np.asarray(contour['points_xy'], float)
    start, end = corners[6]['point_index'], corners[0]['point_index']
    target_dense = points[(start+np.arange((end-start) % len(points)+1)) % len(points)]
    assert all(s['visibility'] == 'observed' for s in contour['segments'])
    result = outline.match_partial(target_dense, matcher=corner, articulation=None)
    assert result.status == 'matched'
    print('dense_same_mask:', result.status, 'score=', result.best['score'])
    print('corner errors:', result.best['metrics'])
    print('per-edge scores:', [e['score'] for e in result.best['edges']])
    print('decision guards:', result.best['limit_ratios'])
    records.append(dict(case='dense_same_mask', evidence='synthetic_raster_mask', result=result.to_dict()))
    if output_dir is not None:
        (output_dir/'results.json').write_text(json.dumps(dict(
            evidence='synthetic_contracts_not_real_accuracy', target_points_xy=target.tolist(), cases=records),
            indent=2, allow_nan=False)+'\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    run_examples(parser.parse_args().output_dir)
