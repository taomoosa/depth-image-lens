"""Conservative, query-only inward positions for visible outer-contour corners."""
from numbers import Real

import numpy as np

from .debug import _opencv
from .outline import _periodic_interp


def _point_location(point, polygon):
    """Return -1 outside, 0 on the boundary, 1 inside, in float64.

    OpenCV's float32 point test would erase small fractional displacements.
    Treat machine-roundoff proximity to an edge conservatively as boundary.
    """
    end = np.roll(polygon, -1, axis=0)
    edge = end - polygon
    relative = point - polygon
    tolerance = 64 * np.finfo(float).eps * max(1., float(np.abs(polygon).max()))
    cross = edge[:, 0] * relative[:, 1] - edge[:, 1] * relative[:, 0]
    on_line = np.abs(cross) <= tolerance * np.linalg.norm(edge, axis=1)
    in_box = np.all((point >= np.minimum(polygon, end) - tolerance)
                    & (point <= np.maximum(polygon, end) + tolerance), axis=1)
    if np.any(on_line & in_box):
        return 0
    straddles = (polygon[:, 1] > point[1]) != (end[:, 1] > point[1])
    a, delta = polygon[straddles], edge[straddles]
    crossings = a[:, 0] + (point[1] - a[:, 1]) * delta[:, 0] / delta[:, 1]
    return 1 if np.count_nonzero(point[0] < crossings) % 2 else -1


def _touches_boundary(origin, direction, distance, polygon):
    """Whether the open-start ray segment touches a pixel-center polygon.

    Work in float64 and check every edge, including collinear overlap. Endpoint
    containment alone would allow jumping across a notch or an internal hole.
    """
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = polygon - origin
    cross = direction[0] * edges[:, 1] - direction[1] * edges[:, 0]
    side = relative[:, 0] * direction[1] - relative[:, 1] * direction[0]
    tolerance = 64 * np.finfo(float).eps * max(1., float(np.abs(polygon).max()))
    parallel = np.abs(cross) <= 64 * np.finfo(float).eps
    if np.any(~parallel):
        along = (relative[:, 0] * edges[:, 1] - relative[:, 1] * edges[:, 0])[~parallel] / cross[~parallel]
        fraction = side[~parallel] / cross[~parallel]
        if np.any((along > tolerance) & (along <= distance + tolerance)
                  & (fraction >= -tolerance) & (fraction <= 1 + tolerance)):
            return True
    collinear = parallel & (np.abs(side) <= tolerance)
    if np.any(collinear):
        a = relative[collinear] @ direction
        b = (relative[collinear] + edges[collinear]) @ direction
        if np.any((np.maximum(a, b) > tolerance)
                  & (np.minimum(a, b) <= distance + tolerance)):
            return True
    return False


def corner_interior_points(result, offset_px, contour_id):
    if isinstance(offset_px, (bool, np.bool_)) or not isinstance(offset_px, Real):
        raise ValueError('offset_px must be a finite nonnegative real number')
    try:
        offset_px = float(offset_px)
    except (OverflowError, ValueError) as error:
        raise ValueError('offset_px must be a finite nonnegative real number') from error
    if not np.isfinite(offset_px) or offset_px < 0:
        raise ValueError('offset_px must be a finite nonnegative real number')
    contours = result._id_map(result.contours, 'contours')
    if contour_id is None:
        largest = result.largest_contour()
        if largest is None:
            return []
        contour_id = largest['id']
    if (isinstance(contour_id, (bool, np.bool_))
            or not isinstance(contour_id, (int, np.integer)) or contour_id not in contours):
        raise ValueError('contour_id must identify an existing contour')
    contour = contours[contour_id]
    corners = result._id_map(contour['corners'], 'corners')
    if not corners:
        return []

    cv = _opencv()
    points = np.asarray(contour['points_xy'], dtype=float)
    holes = [np.asarray(h['points_xy'], dtype=float) for h in result.holes
             if h['parent_contour_id'] == contour_id]
    polygons = [points] + holes
    component = result.component_mask(contour_id)
    radius = result.parameters['clip_margin_px'] + 1
    safe = cv.erode(result.observed_mask.astype(np.uint8),
                    np.ones((2 * radius + 1, 2 * radius + 1), np.uint8),
                    borderType=cv.BORDER_CONSTANT, borderValue=0) > 0
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    positions = np.r_[0., np.cumsum(lengths[:-1])]
    perimeter = float(lengths.sum())
    window = result.parameters['corner_window_px']
    rows = []
    for corner_id, corner in corners.items():
        source = np.asarray(corner['xy'], dtype=float)
        row = dict(contour_id=int(contour_id), corner_id=corner_id,
                   source_xy=source.tolist(), xy=None, sample_xy=None,
                   inward_unit_xy=None, offset_px=offset_px,
                   status='unavailable', reason=None)
        rows.append(row)
        if contour['status'] != 'computed' or contour['orientation'] != 'clockwise_in_image_coordinates':
            row['reason'] = 'ambiguous_or_insufficient_contour'
            continue
        if corner['visibility'] != 'observed':
            row['reason'] = 'corner_not_observed'
            continue
        index = corner['point_index']
        if not np.array_equal(source, points[index]):
            raise ValueError('corner xy must match its raw point_index')
        # Include both endpoints of every raw edge intersecting the support arc,
        # not just the nearest vertex at each chord end.
        forward = np.mod(positions - positions[index], perimeter)
        backward = np.mod(positions[index] - positions, perimeter)
        support = (forward <= window) | (backward <= window)
        support |= np.roll(forward < window, 1) | np.roll(backward < window, -1)
        support_points = points[support].astype(np.intp)
        if not safe[support_points[:, 1], support_points[:, 0]].all():
            row['reason'] = 'unobserved_corner_support'
            continue
        before, after = _periodic_interp(points, positions,
                                         positions[index] + np.array([-window, window]), perimeter)
        incoming, outgoing = source - before, after - source
        norms = np.array([np.linalg.norm(incoming), np.linalg.norm(outgoing)])
        if np.any(norms <= 1e-12):
            row['reason'] = 'degenerate_corner_direction'
            continue
        tangent_sum = incoming / norms[0] + outgoing / norms[1]
        direction = np.array([-tangent_sum[1], tangent_sum[0]])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            row['reason'] = 'degenerate_corner_direction'
            continue
        direction /= norm
        row['inward_unit_xy'] = direction.tolist()
        target = source + offset_px * direction
        height, width = component.shape
        if not (0 <= target[0] <= width - 1 and 0 <= target[1] <= height - 1):
            row['reason'] = 'offset_point_not_inside_component'
            continue
        sample = np.floor(target + .5).astype(np.intp)
        if not component[sample[1], sample[0]]:
            row['reason'] = 'offset_point_not_inside_component'
            continue
        if offset_px > 0:
            if (_point_location(target, points) <= 0
                    or any(_point_location(target, p) >= 0 for p in holes)):
                row['reason'] = 'offset_point_not_inside_component'
                continue
            if any(_touches_boundary(source, direction, offset_px, p) for p in polygons):
                row['reason'] = 'offset_path_touches_boundary'
                continue
        row.update(xy=target.tolist(), sample_xy=sample.tolist(), status='computed', reason=None)
    return rows
