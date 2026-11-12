"""Local depth-validated color hypotheses; no temporal or semantic certainty."""
import numpy as np
from ._rgbd_fusion import _shift, depth_support
from .boundary import _components


SUPPORTED_DECISION_REASONS = {
    0: 'no_selected_change_or_unavailable',
    1: 'color_seed_with_local_depth_support',
    2: 'strong_depth_seed_without_strict_color',
    3: 'color_candidate_without_supported_depth_seed',
    4: 'unselected_raw_depth_candidate',
    5: 'unresolved_local_color_or_missing_evidence',
    6: 'removed_by_min_object_area',
    7: 'bounded_seed_color_growth_inference',
}


def color_supported_fusion(delta, depth_fg, farther, color_fg, color_usable,
                           color_uncertain, lab, edge_valid, color_debug, valid,
                           roi, parameters):
    """Validate seeds with local depth, then grow their *fixed* full-Lab colors.

    Distance to the source seed and bounded path length prevent gradual color
    drift or a single global connected-component vote legitimizing a shadow.
    Near-table pixels may be inferred at the boundary; they are not measured
    foreground. A trustworthy unchanged-background appearance blocks growth.
    Missing depth/color and farther depth block propagation, never get filled.
    """
    p = parameters
    t = p['depth_threshold_m']
    fraction, supported = depth_support(delta, depth_fg, p)
    reliable = color_usable & ~color_uncertain & ~color_debug['background_photometric_clipped_mask']
    strict = reliable & color_fg
    # Division avoids overflowing finite large threshold/ratio products.
    with np.errstate(over='ignore', invalid='ignore'):
        strong = depth_fg & (delta / t > p['supported_anchor_ratio'])
        beyond_margin = delta / t > 1. + p['depth_margin_m'] / t
    support = supported & beyond_margin & edge_valid
    raw_seeds = support & (strong | strict)
    labels, parts = _components(raw_seeds)
    keep = np.zeros(len(parts) + 1, bool)
    for label, area in parts:
        keep[label] = area >= p['supported_min_seed_area']
    seeds = keep[labels]

    # Full-Lab residual includes luminance. This check is local only and does
    # not globally lower the original background subtraction thresholds.
    relaxed = color_usable & (color_debug['background_combined_score'] >
                              p['supported_relaxation_ratio'] ** 2)
    near_table = valid & (np.abs(delta) <= p['depth_margin_m'])
    background_guard = near_table & reliable & ~relaxed
    # An invalid/rejected reference cannot authorize new color-only pixels.
    # Depth-supported source pixels still remain valid hypotheses.
    domain = valid & edge_valid & ~farther & (relaxed | depth_fg) & ~background_guard
    domain |= seeds
    selected = seeds.copy()
    prototype = lab.copy()
    frontier = seeds.copy()
    step_map = np.full(valid.shape, -1, np.int8)
    step_map[seeds] = 0
    limit2 = p['supported_color_tolerance'] ** 2
    for step in range(1, p['supported_growth_radius_px'] + 1):
        best = np.full(valid.shape, np.inf)
        chosen = prototype.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                available = _shift(frontier, dy, dx, False) & domain & ~selected
                source = _shift(prototype, dy, dx)
                distance2 = np.sum((lab - source) ** 2, axis=2)
                local2 = np.sum((lab - _shift(lab, dy, dx)) ** 2, axis=2)
                eligible = available & (distance2 <= limit2) & (local2 <= limit2)
                better = eligible & (distance2 < best)
                best[better] = distance2[better]
                chosen[better] = source[better]
        frontier = np.isfinite(best)
        if not frontier.any():
            break
        prototype[frontier] = chosen[frontier]
        selected |= frontier
        step_map[frontier] = step

    # Every connected selected region must retain a nontrivial measured depth
    # fraction. This is an additional veto, never a whole-region acceptance rule.
    labels, parts = _components(selected)
    measured = np.bincount(labels[support], minlength=len(parts) + 1)
    retain = np.zeros(len(parts) + 1, bool)
    support_ratio = np.zeros(len(parts) + 1, float)
    for label, area in parts:
        support_ratio[label] = measured[label] / area
        retain[label] = support_ratio[label] >= p['supported_min_depth_fraction']
    rejected = selected & ~retain[labels]
    selected &= retain[labels]
    accepted_seeds = seeds & selected
    inferred = selected & ~accepted_seeds
    color_seeds = accepted_seeds & strict
    depth_seeds = accepted_seeds & ~strict
    # No color-only seed is selected. Rejected candidates remain unresolved:
    # same-depth color change can also be a physically thin real object.
    color_without_support = color_fg & ~selected & ~depth_fg
    suppressed = depth_fg & ~selected
    deferred = roi & ~selected & ~suppressed & (color_fg | rejected | (relaxed & ~valid))
    confident_fg = color_seeds & reliable
    confident_bg = background_guard & ~selected
    unknown = roi & (~valid | ~edge_valid | inferred | depth_seeds |
                     suppressed | deferred | rejected)
    decision = np.zeros(valid.shape, np.uint8)
    for code, mask in ((3, color_without_support), (4, suppressed),
                       (5, deferred & ~color_without_support),
                       (1, color_seeds), (2, depth_seeds), (7, inferred)):
        decision[mask] = code
    debug = dict(fusion_decision=decision, depth_support_fraction=fraction,
                 locally_supported_depth_mask=supported, strong_depth_mask=strong,
                 current_color_edge_valid_mask=edge_valid,
                 supported_seed_mask=accepted_seeds, raw_supported_seed_mask=raw_seeds,
                 color_seed_mask=color_seeds, depth_seed_mask=depth_seeds,
                 relaxed_color_mask=relaxed, growth_candidate_mask=domain,
                 growth_step=step_map, accepted_growth_mask=inferred.copy(),
                 accepted_both_mask=color_seeds.copy(),
                 accepted_depth_only_mask=depth_seeds.copy(),
                 accepted_color_only_mask=np.zeros(valid.shape, bool),
                 unsupported_color_mask=color_without_support,
                 support_rejected_mask=rejected,
                 region_depth_support_fraction=support_ratio[labels],
                 background_guard_mask=background_guard,
                 inferred_foreground_mask=inferred,
                 confident_foreground_mask=confident_fg,
                 confident_background_mask=confident_bg, unknown_mask=unknown)
    return selected, suppressed, deferred, debug
