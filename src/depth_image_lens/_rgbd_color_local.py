"""Local signed-depth evidence with RGB-guided boundary inference.

Spatial averages are evidence, not calibrated probabilities. Correlated depth
bias and objects indistinguishable from shadows remain unresolved limitations.
"""
import numpy as np
from ._rgbd_fusion import _shift
from .debug import _opencv

LOCAL_DECISION_REASONS = {
    0: 'no_color_candidate_or_unavailable',
    1: 'local_depth_supported_seed',
    3: 'locally_table_like_candidate',
    4: 'depth_outside_local_selection',
    5: 'local_evidence_unresolved_or_depth_unavailable',
    6: 'removed_by_min_object_area',
    7: 'bounded_seed_color_boundary_inference',
}


def color_local_fusion(delta, depth_fg, farther, color_fg, color_usable,
                       color_uncertain, lab, edge_valid, color_debug, valid,
                       roi, parameters):
    p = parameters
    radius = p['local_support_radius_px']
    usable = color_usable & edge_valid & ~color_debug['background_photometric_clipped_mask']
    candidate = roi & usable & (color_debug['background_combined_score'] >
                               p['local_color_relaxation_ratio'] ** 2)
    upper = np.zeros(valid.shape, bool)
    if p['local_max_reference_separation_m'] is not None:
        upper = valid & (delta > p['local_max_reference_separation_m'])
    # An invalid/high pixel never becomes inferred foreground. Negative samples
    # remain in the statistics: do not average only positive foreground samples.
    sample = candidate & valid & ~upper
    domain = sample & ~farther
    finite_delta = np.where(valid, delta, 0.)
    clipped = np.clip(finite_delta, -p['local_depth_clip_m'], p['local_depth_clip_m'])
    total = np.zeros(valid.shape, float)
    seen = total.copy(); sums = total.copy(); positive = total.copy()
    sigma2 = p['local_color_sigma'] ** 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = _shift(candidate & ~upper, dy, dx, False)
            distance2 = np.sum((lab - _shift(lab, dy, dx)) ** 2, axis=2)
            weight = np.exp(-.5 * distance2 / sigma2) * candidate * neighbor
            # Outside the image/ROI is unavailable, not a negative depth vote.
            total += weight
            observed_weight = weight * _shift(sample, dy, dx, False)
            seen += observed_weight
            sums += observed_weight * _shift(clipped, dy, dx)
            positive += observed_weight * _shift(depth_fg, dy, dx, False)
    coverage = np.divide(seen, total, out=np.zeros_like(seen), where=total > 0)
    mean = np.divide(sums, seen, out=np.zeros_like(seen), where=seen > 0)
    density = np.divide(positive, seen, out=np.zeros_like(seen), where=seen > 0)
    enough = domain & (coverage >= p['local_min_valid_fraction']) & (seen >= 3.)
    background = enough & (mean <= p['local_background_depth_m']) & (density <= p['local_background_max_depth_fraction'])
    raw_seeds = enough & (mean > p['local_mean_depth_m']) & (density >= p['local_min_depth_fraction'])
    cv = _opencv()
    n, labels, stats, _ = cv.connectedComponentsWithStats(raw_seeds.astype(np.uint8), connectivity=8)
    keep = stats[:, cv.CC_STAT_AREA] >= p['local_seed_min_area']; keep[0] = False
    seeds = keep[labels]
    selected = seeds.copy(); frontier = seeds.copy(); prototype = lab.copy()
    steps = np.full(valid.shape, -1, np.int8); steps[seeds] = 0
    grow_domain = domain & ~background & enough
    tolerance2 = p['local_color_tolerance'] ** 2
    for step in range(1, p['local_growth_radius_px'] + 1):
        best = np.full(valid.shape, np.inf); chosen = prototype.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                source = _shift(prototype, dy, dx)
                distance2 = np.sum((lab-source)**2, axis=2)
                local2 = np.sum((lab-_shift(lab, dy, dx))**2, axis=2)
                eligible = (_shift(frontier, dy, dx, False) & grow_domain & ~selected &
                            (distance2 <= tolerance2) & (local2 <= tolerance2))
                better = eligible & (distance2 < best)
                best[better] = distance2[better]; chosen[better] = source[better]
        frontier = np.isfinite(best)
        if not frontier.any():
            break
        selected |= frontier; prototype[frontier] = chosen[frontier]; steps[frontier] = step
    # Even local seeds with subthreshold center depth are inferred, not measured
    # foreground. "Background" is only table-like under this noise model.
    inferred = selected & (~seeds | ~depth_fg)
    confident = seeds & depth_fg & ~color_uncertain
    suppressed = depth_fg & ~selected
    deferred = candidate & ~selected & ~suppressed & ~background
    unknown = roi & (~valid | ~usable | inferred | suppressed | deferred | upper)
    decision = np.zeros(valid.shape, np.uint8)
    decision[suppressed] = 4; decision[deferred | (candidate & ~valid) | upper] = 5
    decision[background] = 3; decision[seeds] = 1; decision[inferred] = 7
    debug = dict(fusion_decision=decision, local_candidate_mask=candidate,
                 local_mean_depth_m=mean, local_depth_support_fraction=density,
                 local_valid_fraction=coverage, local_effective_samples=seen,
                 raw_local_seed_mask=raw_seeds, local_seed_mask=seeds,
                 local_background_mask=background, local_upper_excluded_mask=upper,
                 growth_step=steps, accepted_both_mask=selected & depth_fg,
                 accepted_depth_only_mask=np.zeros(valid.shape,bool),
                 accepted_color_only_mask=np.zeros(valid.shape,bool),
                 accepted_growth_mask=inferred.copy(), inferred_foreground_mask=inferred,
                 confident_foreground_mask=confident, confident_background_mask=background,
                 unknown_mask=unknown)
    return selected, suppressed, deferred, debug
