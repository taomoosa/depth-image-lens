"""Whole-component raster inset, independent of corner/segment classification."""
from math import sqrt
from numbers import Real

import numpy as np

from .debug import _opencv
from .outline import OuterContourInsetResult, _canonical


def _distance_threshold(offset_px):
    # Distances on the integer lattice are square roots of integers. Compare
    # in the gap, so float32 EDT rounding does not change equality at (e.g.)
    # sqrt(2), or conflate a distance just below 1 with 1 itself. At large
    # distances use OpenCV resolution rather than assuming a separable gap.
    if offset_px >= 1024:
        return np.float32(offset_px)
    squared = int(offset_px * offset_px)
    if offset_px < sqrt(squared):
        squared -= 1
    elif offset_px >= sqrt(squared + 1):
        squared += 1
    return np.float32(sqrt(squared + .5))


def _ring(raw, origin, *, hole=False):
    points, area = _canonical(raw[:, 0, :])
    if hole and area:
        # Opposite winding to the shell, preserving the canonical start.
        points = points[np.r_[0, np.arange(len(points) - 1, 0, -1)]]
    points = points + origin
    vertices = set(map(tuple, points.tolist()))
    edge_end = np.roll(points, -1, axis=0)
    diagonal = np.all(np.abs(edge_end - points) == 1, axis=1)
    # Unit 8-neighbor lattice segments can cross away from their vertices only
    # as the two opposite diagonals of the same unit square.
    cells = np.minimum(points[diagonal], edge_end[diagonal])
    direction = (edge_end[diagonal, 0] - points[diagonal, 0]) == (
        edge_end[diagonal, 1] - points[diagonal, 1])
    diagonals = {(int(x), int(y), bool(d)) for (x, y), d in zip(cells, direction)}
    touches = len(vertices) != len(points) or any(
        (x, y, not d) in diagonals for x, y, d in diagonals)
    status = ('degenerate' if len(points) < 3 or not area else
              'self_touch_or_crossing' if touches else 'simple')
    return dict(points_xy=points.tolist(), area_px2=area, status=status), vertices, diagonals


def _trace(mask, origin, cv):
    """Trace a nonempty 8-connected raster selection with CCOMP hole grouping."""
    raw, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    parents = hierarchy[0, :, 3]
    _, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    grouped = {}
    for i, parent in enumerate(parents):
        grouped.setdefault(int(parent), []).append(i)
    rings = [_ring(contour, origin, hole=parents[i] != -1) for i, contour in enumerate(raw)]

    def start(i):
        x, y = rings[i][0]['points_xy'][0]
        return y, x

    parts = []
    for root in sorted(grouped[-1], key=start):
        outer, vertices, diagonals = rings[root]
        holes = []
        reasons = set()
        if outer['status'] != 'simple':
            reasons.add('outer_ring_' + outer['status'])
        for i in sorted(grouped.get(root, []), key=start):
            record, hv, hd = rings[i]
            holes.append(record)
            if record['status'] != 'simple':
                reasons.add('hole_ring_' + record['status'])
            if not vertices.isdisjoint(hv) or any((x, y, not d) in diagonals for x, y, d in hd):
                reasons.add('rings_touch_or_cross')
            vertices.update(hv)
            diagonals.update(hd)
        x, y = raw[root][0, 0]
        label = labels[y, x]
        parts.append(dict(id=len(parts), outer=outer, holes=holes,
                          foreground_pixel_count=int(stats[label, cv.CC_STAT_AREA]),
                          polygon_status='ambiguous' if reasons else 'simple',
                          polygon_reasons=sorted(reasons)))
    return parts


def inset_component(result, offset_px, contour_id):
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
        contour_id = None if largest is None else largest['id']
    elif (isinstance(contour_id, (bool, np.bool_))
          or not isinstance(contour_id, (int, np.integer)) or contour_id not in contours):
        raise ValueError('contour_id must identify an existing contour')

    mask = np.zeros_like(result.object_mask, dtype=bool)
    parts = []
    summary = dict(status='no_visible_component', contour_id=None, offset_px=offset_px,
                   source_contour_status=None, source_contour_clipped=None,
                   source_touches_unobserved_or_frame=None, source_pixel_count=0,
                   inset_pixel_count=0, component_count=0, hole_count=0,
                   polygon_status='unavailable',
                   scope='visible_component_pixels_not_physical_material_boundary')
    if contour_id is None:
        return OuterContourInsetResult(mask, parts, summary)

    cv = _opencv()
    component = result.component_mask(contour_id)
    selected = contours[contour_id]
    x, y, width, height = cv.boundingRect(component.astype(np.uint8))
    box = np.s_[y:y + height, x:x + width]
    local = component[box].astype(np.uint8)
    safe = cv.erode(result.observed_mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                    borderType=cv.BORDER_CONSTANT, borderValue=0)[box]
    summary.update(contour_id=int(contour_id), source_contour_status=selected['status'],
                   source_contour_clipped=bool(selected['clipped']),
                   source_touches_unobserved_or_frame=bool(np.any(local & ~safe.astype(bool))),
                   source_pixel_count=int(local.sum()))

    retained = local
    if offset_px > 0:
        # Every row/column outside the component bbox is excluded. The one-zero
        # padding includes that nearest complement even when the image is full.
        max_distance = (min(width, height) + 1) // 2
        if offset_px >= max_distance:
            retained = np.zeros_like(local)
        else:
            distance = cv.distanceTransform(np.pad(local, 1), cv.DIST_L2,
                                            cv.DIST_MASK_PRECISE)[1:-1, 1:-1]
            retained = (distance > _distance_threshold(offset_px)).astype(np.uint8)
    mask[box] = retained
    if retained.any():
        parts = _trace(retained, np.array([x, y]), cv)
        summary.update(status='computed' if len(parts) == 1 else 'split',
                       inset_pixel_count=int(retained.sum()), component_count=len(parts),
                       hole_count=sum(len(part['holes']) for part in parts),
                       polygon_status=('simple' if all(part['polygon_status'] == 'simple' for part in parts)
                                       else 'ambiguous'))
    else:
        summary['status'] = 'collapsed'
    return OuterContourInsetResult(mask, parts, summary)
