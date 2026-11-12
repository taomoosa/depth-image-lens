"""Whole connected color-region evidence; not per-pixel depth confirmation."""
import numpy as np
from .debug import _opencv

REGION_DECISION_REASONS = {
    0: 'no_color_candidate_or_unavailable',
    1: 'region_accepted_raw_depth_supported_pixel',
    3: 'region_insufficient_depth_support',
    4: 'raw_depth_outside_selected_color_region',
    5: 'region_unobserved_small_or_color_budget_exceeded',
    6: 'removed_by_min_object_area',
    7: 'region_accepted_inferred_pixel',
}


def color_region_fusion(delta, depth_fg, farther, color_fg, color_usable,
                        color_uncertain, lab, edge_valid, color_debug, valid,
                        roi, parameters):
    """Partition full Lab colors, then judge each 8-connected region as a whole.

    Fixed representative colors prevent transitive color drift. Each region is
    tested by observed coverage, a winsorized signed depth mean, AND foreground
    density. Raw depth remains untouched. This intentionally allows support to
    extend through an entire region; identical-color touching shadows/holes are
    fundamentally dangerous, even when all numeric tests pass.
    """
    p = parameters
    candidate = (roi & color_usable & edge_valid &
                 (color_debug['background_combined_score'] > p['region_color_relaxation_ratio']**2))
    shape = valid.shape
    color_ids = np.zeros(shape, np.int32)
    if candidate.any():
        colors, inverse, counts = np.unique(lab[candidate], axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        assignments = np.zeros(len(colors), np.int32)
        for group in range(1, p['region_max_color_groups'] + 1):
            remaining = assignments == 0
            if not remaining.any():
                break
            seed = int(np.argmax(np.where(remaining, counts, 0)))
            near = remaining & (np.linalg.norm(colors - colors[seed], axis=1) <= p['region_color_tolerance'])
            assignments[near] = group
        color_ids[candidate] = assignments[inverse]
    region_ids = np.zeros(shape, np.int32)
    selected = np.zeros(shape, bool)
    stats = []
    support_map = np.zeros(shape, float)
    coverage_map = np.zeros(shape, float)
    mean_map = np.full(shape, np.nan)
    small_or_missing = candidate & (color_ids == 0)
    next_id = 0
    t = p['depth_threshold_m']
    for group in range(1, int(color_ids.max()) + 1):
        # Component bounding boxes avoid scanning the entire image per speckle.
        cv = _opencv()
        count, labels, boxes, _ = cv.connectedComponentsWithStats(
            (color_ids == group).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            area = int(boxes[label, cv.CC_STAT_AREA])
            next_id += 1
            x, y, w, h = boxes[label, :4]
            sl = np.s_[y:y+h, x:x+w]
            part = labels[sl] == label
            region_ids[sl][part] = next_id
            seen = part & valid[sl]
            n = int(seen.sum())
            coverage = n / area
            density = float(np.count_nonzero(depth_fg[sl] & seen) / n) if n else 0.
            # Clip dimensionless ratios before averaging: isolated enormous
            # sensor errors cannot dominate a region mean. No Gaussian/iid
            # probability is claimed for these correlated pixel measurements.
            with np.errstate(over='ignore', invalid='ignore'):
                ratios = delta[sl][seen] / t
            mean_ratio = float(np.clip(ratios, -p['region_depth_clip_ratio'], p['region_depth_clip_ratio']).mean()) if n else None
            enough = area >= p['region_min_area'] and coverage >= p['region_min_valid_fraction']
            accepted = bool(enough and mean_ratio is not None and mean_ratio > p['region_mean_depth_ratio'] and density >= p['region_min_depth_fraction'])
            selected[sl] |= seen & ~farther[sl] & accepted
            if not enough:
                small_or_missing[sl] |= part
            support_map[sl][part] = density
            coverage_map[sl][part] = coverage
            if mean_ratio is not None:
                mean_map[sl][part] = mean_ratio * t
            stats.append(dict(id=next_id, color_group=group, area=area,
                              valid_pixels=n, valid_fraction=coverage,
                              depth_support_fraction=density, clipped_mean_depth_ratio=mean_ratio,
                              accepted=accepted))
    inferred = selected & ~depth_fg
    suppressed = depth_fg & ~selected
    deferred = candidate & ~selected & ~suppressed
    # All inferred foreground remains unknown; absence of depth support is not
    # proof of background, so rejected color changes remain unresolved too.
    unknown = roi & (~valid | ~edge_valid | inferred | suppressed | deferred)
    decision = np.zeros(shape, np.uint8)
    decision[deferred] = 3
    decision[suppressed] = 4
    decision[small_or_missing] = 5
    decision[selected & depth_fg] = 1
    decision[inferred] = 7
    debug = dict(fusion_decision=decision, region_label_map=region_ids,
                 region_color_group_map=color_ids, region_candidate_mask=candidate,
                 region_depth_support_fraction=support_map,
                 region_valid_fraction=coverage_map, region_clipped_mean_depth_m=mean_map,
                 region_statistics=stats, region_color_budget_exceeded=bool((candidate & (color_ids == 0)).any()),
                 accepted_both_mask=selected & depth_fg, accepted_depth_only_mask=np.zeros(shape,bool),
                 accepted_color_only_mask=np.zeros(shape,bool), accepted_growth_mask=inferred,
                 inferred_foreground_mask=inferred, confident_foreground_mask=selected & depth_fg & ~color_uncertain,
                 confident_background_mask=np.zeros(shape,bool), unknown_mask=unknown)
    return selected, suppressed, deferred, debug
