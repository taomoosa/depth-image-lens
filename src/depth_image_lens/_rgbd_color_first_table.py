"""Optional color-first table calibration from a caller's spatial anchors."""
from numbers import Integral
from math import ceil

import numpy as np
from .debug import _opencv


def table_points(value):
    """Normalize JSON arrays or Python sequences to unique integer XY pairs."""
    if value is None:
        return None
    try:
        if not 2 <= len(value) <= 12:
            raise ValueError
        points = []
        for xy in value:
            if len(xy) != 2 or any(isinstance(v, (bool, np.bool_)) or
                                  not isinstance(v, Integral) or int(v) < 0 for v in xy):
                raise ValueError
            points.append(tuple(int(v) for v in xy))
        if len(set(points)) != len(points):
            raise ValueError
        return tuple(points)
    except (TypeError, ValueError) as exc:
        raise ValueError('first_table_xy must be None or 2..12 distinct nonnegative integer XY pairs') from exc


def select_table(delta, valid, usable, lab, reference_chroma, p):
    """Vote across full usable RGB; never treat anchor pixels as depth samples."""
    cv = _opencv()
    anchors = p['first_table_xy']
    shape = lab.shape[:2]
    zero = np.zeros(shape, bool)
    report = dict(status='insufficient_anchors', method=p['first_table_statistic'],
                  anchors=len(anchors), anchor_xy=[list(xy) for xy in anchors],
                  usable_anchors=0, votes=0, offset_m=None)
    points = [(x, y) for x, y in anchors if 0 <= x < shape[1] and 0 <= y < shape[0] and usable[y, x]]
    report['usable_anchors'] = len(points)
    quorum = len(anchors)//2 + 1
    if len(points) < quorum:
        return zero, zero, zero, zero, report
    scales = np.array([1/p['model_luma_tolerance'], 1/p['model_chroma_tolerance'], 1/p['model_chroma_tolerance']])
    coordinates = np.asarray(points)
    xx, yy = coordinates.T
    colors = lab[yy, xx]
    hypotheses = {}
    competing = set()
    for color in np.unique(colors, axis=0):
        matched = np.flatnonzero(np.linalg.norm((colors-color)*scales, axis=1) <= 1)
        if len(matched) < 2:
            continue
        color_mask = usable & (np.sum(((lab-color)*scales)**2, axis=2) <= 1)
        count, labels, stats, _ = cv.connectedComponentsWithStats(color_mask.astype(np.uint8), connectivity=8)
        for label in np.unique(labels[yy, xx]):
            if not label or label >= count:
                continue
            votes = tuple(np.flatnonzero(labels[yy, xx] == label))
            if len(votes) < 2:
                continue
            competing.add(votes)
            if len(votes) < quorum:
                continue
            proposal = labels == label
            if votes in hypotheses:
                # Same votes can yield different prototype extents; their
                # common connected core is the conservative choice.
                hypotheses[votes] &= proposal
            else:
                hypotheses[votes] = proposal
    if not hypotheses:
        report['status'] = 'no_shared_color_region'
        return zero, zero, zero, zero, report
    best = max(hypotheses, key=len)
    if any(set(other).isdisjoint(best) for other in competing) or sum(
            len(votes) == len(best) for votes in hypotheses) > 1:
        report['status'] = 'ambiguous_color_regions'
        return zero, zero, zero, zero, report
    component = hypotheses[best]
    _, labels = cv.connectedComponents(component.astype(np.uint8), connectivity=8)
    common_labels = labels[yy[list(best)], xx[list(best)]]
    if not common_labels[0] or not np.all(common_labels == common_labels[0]):
        report['status'] = 'no_shared_color_region'
        return zero, zero, zero, zero, report
    component = labels == common_labels[0]
    voters = coordinates[list(best)].astype(float)
    span = float(np.sqrt(np.max(np.sum((voters[:,None]-voters[None,:])**2,axis=2))))
    area = int(component.sum())
    report.update(votes=len(best), anchor_span_px=span, proposal_area=area,
                  voted_xy=[list(points[i]) for i in best])
    if span < p['first_table_min_anchor_span_px']:
        report['status'] = 'anchors_too_close'
        return component, zero, zero, zero, report
    if area < p['first_min_area']:
        report['status'] = 'insufficient_table_area'
        return component, zero, zero, zero, report
    # Replicate the existing color-first inset, excluding nearby color edges
    # from depth calibration. Anchor locations are not the depth sample set.
    inset = p['model_training_inset_px']
    core = cv.erode(component.astype(np.uint8), np.ones((2*inset+1,2*inset+1),np.uint8),
                    borderType=cv.BORDER_CONSTANT, borderValue=1).astype(bool)
    # Do not allow a conspicuous hue change to redefine the table when
    # all user-declared anchors are actually covered. Luminance may change.
    chroma_values = reference_chroma[core]
    anchor_chroma = reference_chroma[yy[list(best)], xx[list(best)]]
    if not len(chroma_values) or not np.all(np.isfinite(chroma_values)):
        report['status'] = 'unavailable_table_chroma'
        return component, zero, zero, zero, report
    chroma_p90 = float(np.quantile(chroma_values, .9))
    report['reference_chroma_p90'] = chroma_p90
    if (chroma_p90 > p['background_chroma_tolerance'] or
            np.any(anchor_chroma > p['background_chroma_tolerance'])):
        report['status'] = 'table_chroma_changed'
        return component, zero, zero, zero, report
    samples = core & valid
    n = int(samples.sum())
    coverage = n/max(1, int(core.sum()))
    report.update(valid_pixels=n, valid_fraction=coverage)
    size = p['first_block_size_px']
    minimum = max(4, ceil(max(4, size*size//2)*p['first_min_valid_fraction']))
    if n < minimum*p['first_min_blocks'] or coverage < p['first_min_valid_fraction']:
        report['status'] = 'insufficient_valid_depth'
        return component, zero, zero, zero, report
    values = delta[samples]
    with np.errstate(over='ignore', invalid='ignore'):
        median = float(np.median(values))
        mean = float(np.mean(values))
    if not np.isfinite(mean):
        scale = float(np.max(np.abs(values)))
        mean = float(np.mean(values/scale)*scale) if scale else 0.
    offset = median if p['first_table_statistic'] == 'median' else mean
    report.update(sample_median_m=median if np.isfinite(median) else None,
                  sample_mean_m=mean if np.isfinite(mean) else None)
    if not np.isfinite(median) or not np.isfinite(offset) or max(
            abs(median), abs(offset)) > p['first_table_max_offset_m']:
        report['status'] = 'offset_out_of_range'
        return component, zero, zero, zero, report
    spread = float(np.quantile(np.abs(values-median), .9))
    report['sample_p90_deviation_m'] = spread
    limit = p['first_table_max_deviation_m']
    if p['first_table_statistic'] == 'mean' and abs(mean-median) > limit:
        report['status'] = 'mean_disagrees_with_median'
        return component, zero, zero, zero, report
    # Tile medians against a single offset guard spatial mismatch. For each
    # voter, additionally require nearby agreement from a patch, never the
    # exact anchor depth. Reject rather than try alternative depth-fit winners.
    ly,lx = np.nonzero(samples)
    tile = max(2, size*2)
    ids = (ly//tile)*((shape[1]+tile-1)//tile)+lx//tile
    order = np.argsort(ids)
    _, starts = np.unique(ids[order], return_index=True)
    tile_values = [float(np.median(values[idx])) for idx in np.split(order, starts[1:]) if len(idx) >= minimum]
    report['depth_tiles'] = len(tile_values)
    tile_deviation = max((abs(v-median) for v in tile_values), default=None)
    report['max_tile_deviation_m'] = tile_deviation
    report['tile_p90_deviation_m'] = (float(np.quantile(np.abs(np.asarray(tile_values)-median), .9))
                                     if tile_values else None)
    if (len(tile_values) < p['first_min_blocks'] or spread > limit or
            tile_deviation > limit):
        report['status'] = 'inconsistent_table_depth'
        return component, zero, zero, zero, report
    for x, y in voters.astype(int):
        sl = np.s_[max(y-size,0):y+size+1, max(x-size,0):x+size+1]
        patch = samples[sl]
        if (patch.sum() < minimum or abs(float(np.median(delta[sl][patch]))-median) >
                min(limit, p['first_min_depth_delta_m'])):
            report['status'] = 'inconsistent_anchor_neighborhood'
            return component, zero, zero, zero, report
    # Keep invalid and strongly displaced pixels out. The tighter, independent
    # table-classification tube must not erase a 1 mm elevated candidate just
    # because its color was similar. The remaining table is reference only.
    close = valid & (np.abs(delta-median) <= limit)
    reference = core & close
    force = component & valid & (np.abs(delta-median) <= min(limit, p['first_min_depth_delta_m']*.5))
    report.update(status='accepted', offset_m=offset,
                  reference_pixels=int(reference.sum()), forced_background_pixels=int(force.sum()))
    return component, core, reference, force, report
