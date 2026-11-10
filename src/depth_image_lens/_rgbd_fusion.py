"""Deterministic evidence gates, not calibrated object probabilities."""
import numpy as np


DECISION_REASONS = {
    0: 'no_selected_change_or_unavailable',
    1: 'depth_and_reliable_color',
    2: 'strong_locally_supported_depth_without_color_confirmation',
    3: 'chroma_only_candidate',
    4: 'depth_suppressed_by_background_color_and_insufficient_strength_or_local_support',
    5: 'unresolved_candidate',
    6: 'removed_by_min_object_area',
}


def depth_support(delta, depth_fg, parameters):
    """Full-square, center-excluded support; missing/outside positions count zero."""
    p = parameters
    radius = p['depth_support_radius_px']
    h, w = delta.shape
    counts = np.zeros(delta.shape, np.uint16)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dy == 0 and dx == 0) or abs(dy) >= h or abs(dx) >= w:
                continue
            target = (slice(max(0, -dy), min(h, h - dy)),
                      slice(max(0, -dx), min(w, w - dx)))
            neighbor = (slice(max(0, dy), min(h, h + dy)),
                        slice(max(0, dx), min(w, w + dx)))
            with np.errstate(over='ignore', invalid='ignore'):
                agreement = np.abs(delta[target] - delta[neighbor]) <= p['depth_support_tolerance_m']
            counts[target] += depth_fg[target] & depth_fg[neighbor] & agreement
    fraction = counts.astype(float) / ((2 * radius + 1) ** 2 - 1)
    supported = depth_fg & (fraction >= p['depth_support_fraction'])
    return fraction, supported


def evidence_fusion(delta, depth_fg, farther, color_fg, color_usable,
                    color_uncertain, chroma_distance, valid, parameters):
    """Return selection and inspectable, mutually exclusive decision reasons.

    Support is the fraction of all neighboring positions (excluding the center)
    with foreground depth difference within tolerance of the center difference.
    Missing/outside-ROI/outside-image positions contribute zero support. Only
    O(HW) temporary storage is used; the bounded radius controls work.
    """
    p = parameters
    fraction, supported = depth_support(delta, depth_fg, p)
    with np.errstate(over='ignore', invalid='ignore'):
        strong = depth_fg & (delta / p['depth_threshold_m'] >= p['strong_depth_ratio'])
        chroma = chroma_distance / p['background_chroma_tolerance'] > 1 + p['background_score_margin']
    reliable = color_usable & ~color_uncertain
    color_confirmation = reliable & color_fg
    background_agreement = reliable & ~color_fg
    both = depth_fg & color_confirmation
    depth_only = depth_fg & ~both & strong & supported
    color_only = ~depth_fg & ~farther & color_confirmation & chroma
    if p['missing_depth_policy'] == 'unknown':
        color_only &= valid
    selected = both | depth_only | color_only
    suppressed = depth_fg & ~selected & background_agreement
    deferred = (depth_fg | color_fg) & ~selected & ~suppressed
    decision = np.zeros(delta.shape, np.uint8)
    for code, mask in ((1, both), (2, depth_only), (3, color_only),
                       (4, suppressed), (5, deferred)):
        decision[mask] = code
    debug = dict(fusion_decision=decision, depth_support_fraction=fraction,
                 locally_supported_depth_mask=supported, strong_depth_mask=strong,
                 reliable_color_foreground_mask=color_confirmation,
                 reliable_color_background_mask=background_agreement,
                 accepted_both_mask=both, accepted_depth_only_mask=depth_only,
                 accepted_color_only_mask=color_only)
    return selected, suppressed, deferred, debug


SEEDED_DECISION_REASONS = {
    0: 'no_selected_change_or_unavailable',
    1: 'strict_chroma_seed_with_depth',
    3: 'strict_chroma_seed_without_depth_confirmation',
    4: 'depth_suppressed_without_seed_or_growth_evidence',
    5: 'unresolved_color_or_depth_candidate',
    6: 'removed_by_min_object_area',
    7: 'bounded_relaxed_color_and_supported_depth_growth',
}


def _neighbors8(mask):
    """Union of in-frame 8-neighbors, without wraparound or border reflection."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            out[max(0, -dy):min(h, h-dy), max(0, -dx):min(w, w-dx)] |= mask[
                max(0, dy):min(h, h+dy), max(0, dx):min(w, w+dx)]
    return out


def color_seeded_fusion(delta, depth_fg, farther, color_fg, color_usable,
                        color_uncertain, color_debug, valid, parameters):
    """Strict chroma seeds plus bounded, depth-supported relaxed-color growth.

    Every added pixel is an inference, not measured material truth. This is a
    deliberately color-guided alternative: it never falls back to depth-only
    components, fills missing depth, or forces a hole/topology to disappear.
    """
    p = parameters
    fraction, supported = depth_support(delta, depth_fg, p)
    usable = color_usable & ~color_debug['background_photometric_clipped_mask']
    with np.errstate(over='ignore', invalid='ignore'):
        chroma = color_debug['background_chroma_distance'] / p['background_chroma_tolerance']
    strict = usable & ~color_uncertain & (chroma > 1 + p['background_score_margin'])
    seeds = strict & ~farther
    if p['missing_depth_policy'] == 'unknown':
        seeds &= valid
    relaxed = usable & (color_debug['background_combined_score'] > p['color_relaxation_ratio'] ** 2)
    candidates = relaxed & supported & ~farther
    selected = seeds.copy()
    step_map = np.full(delta.shape, -1, np.int8)
    step_map[seeds] = 0
    frontier = seeds.copy()
    for step in range(1, p['color_growth_max_steps'] + 1):
        frontier = _neighbors8(frontier) & candidates & ~selected
        if not frontier.any():
            break
        selected |= frontier
        step_map[frontier] = step
    grown = selected & ~seeds
    both = seeds & depth_fg
    color_only = seeds & ~depth_fg
    suppressed = depth_fg & ~selected & usable & ~color_uncertain & ~color_fg
    deferred = (depth_fg | color_fg | relaxed) & ~selected & ~suppressed
    decision = np.zeros(delta.shape, np.uint8)
    for code, mask in ((1, both), (3, color_only), (4, suppressed), (5, deferred), (7, grown)):
        decision[mask] = code
    debug = dict(fusion_decision=decision, depth_support_fraction=fraction,
                 locally_supported_depth_mask=supported, color_seed_mask=seeds,
                 relaxed_color_mask=relaxed, growth_candidate_mask=candidates,
                 growth_step=step_map, accepted_growth_mask=grown,
                 accepted_both_mask=both, accepted_color_only_mask=color_only,
                 accepted_depth_only_mask=np.zeros(delta.shape, bool))
    return selected, suppressed, deferred, debug
