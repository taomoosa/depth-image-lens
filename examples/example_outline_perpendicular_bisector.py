"""Intersect an input segment's perpendicular bisector with the largest outline."""
import argparse
import json

import numpy as np

from depth_image_lens import OuterContourDetector


def perpendicular_bisector_intersections(points_xy, a_xy, b_xy):
    """Intersect an infinite line and a cyclic sequence of XY pixel centers.

    Positive distances run along (-dy, dx) for the segment vector B - A.
    Isolated hits are returned once; collinear portions become closed intervals.
    """
    a, b = np.asarray(a_xy, dtype=float), np.asarray(b_xy, dtype=float)
    points = np.asarray(points_xy, dtype=float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError('segment endpoints must be finite XY pairs')
    if points.shape == (0,):
        points = points.reshape(0, 2)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('contour points must be finite Nx2 XY coordinates')

    with np.errstate(over='ignore', invalid='ignore'):
        delta = b - a
    if not np.isfinite(delta).all():
        raise ValueError('segment coordinates are too large to measure')
    scale = float(np.max(np.abs(delta)))
    if scale == 0:
        raise ValueError('segment endpoints must be distinct')
    axis = delta / scale
    axis /= np.hypot(*axis)
    direction = np.array([-axis[1], axis[0]])
    midpoint = .5 * a + .5 * b
    # The absolute floor is tiny in image pixels. The scaled term allows for
    # floating-point arithmetic on translated/large coordinates.
    coord_scale = max(np.abs(points).max(initial=0.), np.abs(midpoint).max())
    tolerance = max(1e-9, 16 * np.finfo(float).eps * coord_scale)

    with np.errstate(over='ignore', invalid='ignore'):
        relative = points - midpoint
        side, along = relative @ axis, relative @ direction
    if not np.isfinite(side).all() or not np.isfinite(along).all():
        raise ValueError('contour and segment coordinates are too far apart to measure')

    on_line = np.abs(side) <= tolerance
    hits, intervals = [], []
    for i in range(len(points)):
        j = (i + 1) % len(points)
        if on_line[i] and on_line[j]:
            lo, hi = sorted((float(along[i]), float(along[j])))
            if hi - lo <= tolerance:
                hits.append(lo)
            else:
                intervals.append((lo, hi))
        elif on_line[i] or on_line[j]:
            hits.append(float(along[i if on_line[i] else j]))
        elif (side[i] < 0) != (side[j] < 0):
            # Opposite sides: interpolate along the edge. Scale before addition
            # so a (hypothetically) large signed separation does not overflow.
            di, dj = abs(float(side[i])), abs(float(side[j]))
            large = max(di, dj)
            fraction = (di / large) / (di / large + dj / large)
            hits.append((1 - fraction) * float(along[i]) + fraction * float(along[j]))

    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))

    unique, interval_index = [], 0
    for distance in sorted(hits):
        while interval_index < len(merged) and merged[interval_index][1] + tolerance < distance:
            interval_index += 1
        if interval_index < len(merged) and merged[interval_index][0] - tolerance <= distance:
            continue  # The closed overlap interval already contains this hit.
        if not unique or distance - unique[-1] > tolerance:
            unique.append(distance)

    def position(distance):
        with np.errstate(over='ignore', invalid='ignore'):
            point = midpoint + distance * direction
        if not np.isfinite(point).all():
            raise ValueError('intersection could not be represented by finite XY coordinates')
        return dict(xy=point.tolist(), signed_distance_px=distance, distance_px=abs(distance))

    return dict(midpoint_xy=midpoint.tolist(), bisector_direction_xy=direction.tolist(),
                intersections=[position(distance) for distance in unique],
                overlaps=[dict(start=position(lo), end=position(hi)) for lo, hi in merged])


def largest_contour_bisector(mask, a_xy, b_xy):
    """Use the detector's existing largest-polygon-area selection."""
    outline = OuterContourDetector().detect(mask)
    largest = outline.largest_contour()
    result = perpendicular_bisector_intersections(
        [] if largest is None else largest['points_xy'], a_xy, b_xy)
    status = ('no_visible_contour' if largest is None else 'intersected'
              if result['intersections'] or result['overlaps'] else 'no_intersection')
    return dict(status=status, contour_id=None if largest is None else largest['id'], **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', help='Optional bool HxW .npy mask; omit for a synthetic U shape')
    parser.add_argument('--a', type=float, nargs=2, metavar=('X', 'Y'), default=(75, 30))
    parser.add_argument('--b', type=float, nargs=2, metavar=('X', 'Y'), default=(75, 70))
    args = parser.parse_args()
    try:
        if args.mask:
            mask = np.load(args.mask, allow_pickle=False)
        else:
            mask = np.zeros((150, 150), dtype=bool)
            mask[20:121, 20:131] = True
            mask[20:86, 55:95] = False  # the cross-section hits both arms
        result = largest_contour_bisector(mask, args.a, args.b)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
