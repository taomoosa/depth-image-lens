"""Keep raw corners and query separate inward-offset XY positions in pixels.

Run from the repository root after installation:
    python examples/example_outline_interior.py --offset-px 10
    python examples/example_outline_interior.py --shape rectangle --offset-px 10
    python examples/example_outline_interior.py --shape hole --offset-px 110
    python examples/example_outline_interior.py --offset-px 300

"Inward" means into the object material for both convex and concave corners,
not toward the component centroid. The distance is from the raw corner, not
perpendicular clearance from each edge: a right-angle 10px offset moves about
7.071px along each axis. These are processing-image pixels, not millimetres.
The hole example rejects a path across a hole even if its endpoint is foreground;
an oversized offset is also unavailable, never silently shortened.
"""
import argparse
import json

import numpy as np

from depth_image_lens import OuterContourDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset-px', type=float, default=10.,
                        help='Euclidean displacement from each raw corner, in processing pixels')
    parser.add_argument('--shape', choices=('l-shape', 'rectangle', 'hole'), default='l-shape',
                        help='L shape includes a concave corner; hole demonstrates blocked paths')
    args = parser.parse_args()
    if not np.isfinite(args.offset_px) or args.offset_px < 0:
        parser.error('--offset-px must be finite and nonnegative')
    if args.shape == 'hole':
        mask = np.zeros((180, 180), bool)
        mask[20:161, 20:161] = True
        mask[50:80, 50:80] = False
    else:
        mask = np.zeros((160, 190), bool)
        mask[30:121, 35:156] = True
        if args.shape == 'l-shape':
            mask[70:121, 95:156] = False  # concavity: inward is away from the notch
    result = OuterContourDetector().detect(mask)
    largest = result.largest_contour()
    if largest is None:  # also handle an empty mask when adapting this example
        print('[]')
        return
    offsets = {row['corner_id']: row for row in
               result.corner_interior_points(args.offset_px, largest['id'])}
    records = []
    for entry in result.resolve_sequence(largest['id']):
        if entry['type'] != 'corner':
            continue
        record = entry['record']
        inward = offsets[record['id']]
        records.append(dict(corner_id=record['id'], kind=record['kind'],
                            raw_xy=record['xy'], interior=inward))
        if inward['status'] == 'computed':
            x, y = inward['sample_xy']
            assert mask[y, x]  # arrays use YX, returned coordinates use XY
            delta = np.asarray(inward['xy']) - record['xy']
            records[-1]['delta_xy'] = delta.tolist()
            records[-1]['displacement_px'] = float(np.linalg.norm(delta))
            assert np.isclose(records[-1]['displacement_px'], args.offset_px)
            # Use xy for unrounded geometry; sample_xy is only for image indexing.
        # Unavailable rows have xy/sample_xy=None. Check status before sampling.
        # Hole boundaries do not contribute corner rows. A zero offset keeps the
        # raw boundary position and is not a strictly interior point.
    print(json.dumps(records, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
