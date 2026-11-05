"""Bounded current-corner skips: same inputs, limits 0/1/2, measured runtimes.

PYTHONPATH=src python examples/example_corner_outline_skips.py --repeats 3
Optional --output-dir /existing/parent/new-folder saves JSON and debug PNGs.
Synthetic cases are contract checks, not a real-shirt accuracy benchmark.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
from depth_image_lens import CornerOutlineMatcher, OutlineArticulation


def bump(points, edge_index, amplitude):
    delta = points[edge_index+1]-points[edge_index]
    normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
    return np.insert(points, edge_index+1,
                     (points[edge_index]+points[edge_index+1])/2+amplitude*normal, axis=0)


def cases():
    target = np.array([[60,155], [60,70], [30,85], [20,60], [65,30], [90,35]], float)
    current = np.vstack([target, [[110,35], [135,30], [180,60], [170,85], [140,70], [140,155]]])
    art = OutlineArticulation(2, 3, (62.5,50))
    regular = [
        ('clean', current, ['matched']*3),
        ('one_small_corner', bump(current, 1, 3.2), ['not_matched','matched','matched']),
        ('two_small_corners', bump(bump(current, 2, 2.6), 1, 3.2), ['not_matched','not_matched','matched']),
        ('two_endpoint_tails', bump(bump(current, 4, 2.6), 0, 8.), ['not_matched','not_matched','matched']),
        ('large_fold', bump(current, 1, 12.), ['not_matched']*3),
        ('missing_corner', np.delete(current, 2, axis=0), ['not_matched']*3),
        ('large_extra_corners', np.vstack([current[:2], [[48,58],[40,75]], current[2:]]), ['not_matched']*3),
        ('shear', current@np.array([[1., .5], [0, 1.]]).T, ['not_matched']*3),
    ]
    records = [dict(name=name, target=target, current=q, articulation=art, parameters={}, expected=expected)
               for name, q, expected in regular]
    angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
    radii = np.where(np.arange(48) % 2, 80., 100.)
    star = np.c_[radii*np.cos(angles), radii*np.sin(angles)]
    records.append(dict(name='stress_48_corners', target=star[:8], current=star, articulation=None,
                        parameters=dict(fraction_range=(.05,.35)), expected=['ambiguous']*3))
    records.append(dict(name='stress_budget_1024', target=star[:8], current=star, articulation=None,
                        parameters=dict(fraction_range=(.05,.35), max_hypotheses=1024),
                        expected=['ambiguous','ambiguous','unknown']))
    return records


def run_examples(*, repeats=3, output_dir=None):
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError('repeats must be 1..10')
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=False)
    report = []
    for case in cases():
        for limit in (0, 1, 2):
            matcher = CornerOutlineMatcher(max_skipped_corners=limit, **case['parameters'])
            elapsed = []
            for _ in range(repeats):
                start = time.perf_counter()
                result = matcher.match_contour(case['target'], case['current'], articulation=case['articulation'])
                elapsed.append(time.perf_counter()-start)
            assert result.status == case['expected'][limit], (case['name'], limit, result.status)
            b = result.best
            print(f"{case['name']:23s} limit={limit} {result.status:12s} "
                  f"median={statistics.median(elapsed):.4f}s "
                  f"hypotheses={result.diagnostics['evaluated_hypotheses']:4d} "
                  f"skipped={None if b is None else b['skipped_corner_count']} "
                  f"score={None if b is None else round(b['score'],3)}")
            report.append(dict(case=case['name'], max_skipped_corners=limit,
                               median_seconds=statistics.median(elapsed), seconds=elapsed,
                               result=result.to_dict()))
            if output_dir is not None and not case['name'].startswith('stress'):
                image = np.full((240,300,3), 25, np.uint8)
                cv2.fillPoly(image, [np.rint(case['current']).astype(np.int32)], (75,75,75))
                result.save_debug(output_dir/f"{case['name']}-{limit}.png", image)
    if output_dir is not None:
        (output_dir/'results.json').write_text(json.dumps(dict(
            evidence='synthetic_contracts_and_timings_not_real_accuracy', cases=report), indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run_examples(repeats=args.repeats, output_dir=args.output_dir)
