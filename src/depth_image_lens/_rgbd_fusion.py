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


ANCHORED_DECISION_REASONS = {
    0: 'no_selected_change_or_unavailable',
    1: 'supported_depth_anchor',
    2: 'directional_thin_depth_anchor',
    3: 'supported_background_anchor',
    4: 'unselected_raw_depth_candidate_unresolved',
    5: 'unresolved_color_or_boundary_candidate',
    6: 'removed_by_min_object_area',
    7: 'color_edge_refined_foreground_inference',
}


def _shift(array, dy, dx, fill=0):
    """Neighbor lookup with explicit outside value, never wrapping the image."""
    out = np.full_like(array, fill)
    h, w = array.shape[:2]
    if abs(dy) < h and abs(dx) < w:
        out[max(0, -dy):min(h, h-dy), max(0, -dx):min(w, w-dx)] = array[
            max(0, dy):min(h, h+dy), max(0, dx):min(w, w+dx)]
    return out


def _erode(mask, steps):
    out = mask.copy()
    for _ in range(steps):
        out = np.logical_and.reduce([_shift(out, dy, dx, False)
                                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    return out


def depth_anchored_fusion(delta, depth_fg, farther, color_fg, color_usable,
                          color_uncertain, lab, edge_valid, color_debug, valid,
                          roi, parameters):
    """Depth anchors and bounded full-Lab diffusion refinement, not probabilities.

    A color-compatible 3x3 median rejects isolated noise without blurring a
    visible one-pixel feature across its RGB edge. A separate directional raw
    depth path protects supported thin structures even without a color edge.
    Bounded Jacobi relaxation is NOT an exact graph cut and does not guarantee
    convergence within the budget. Spatial support does not verify material.
    """
    p = parameters
    t = p['depth_threshold_m']
    sigma = p['anchored_color_sigma']
    # Work with clipped dimensionless evidence to keep unusual finite input
    # scales and arithmetic safe. Raw metric evidence is returned unchanged.
    with np.errstate(over='ignore', invalid='ignore'):
        d = np.clip(delta / t, -4., 4.)
    neighbors = []
    strong_neighbors = np.zeros(valid.shape, np.uint8)
    with np.errstate(over='ignore', invalid='ignore'):
        strong = valid & (delta / t > p['anchored_anchor_ratio'])
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nv = _shift(valid, dy, dx, False)
            nd = _shift(d, dy, dx, np.nan)
            nc = _shift(lab, dy, dx)
            both_color = edge_valid & _shift(edge_valid, dy, dx, False)
            distance = np.sqrt(np.sum((lab - nc) ** 2, axis=2))
            compatible = ~both_color | (distance <= sigma * 2)
            neighbors.append(np.where(nv & compatible, nd, np.nan))
            if dy or dx:
                strong_neighbors += _shift(strong, dy, dx, False)
    # Every valid center includes itself. Invalid centers are unused but use a
    # finite placeholder to avoid all-NaN median warnings at sensor/ROI gaps.
    stack = np.stack(neighbors)
    stack[:, ~valid] = 0.
    robust = np.nanmedian(stack, axis=0)
    coarse = valid & ~farther & (robust > 1.)
    reliable_reference = color_usable & ~color_uncertain & ~color_debug['background_photometric_clipped_mask']
    reference_agrees = reliable_reference & (color_debug['background_combined_score'] < .25)
    # Spatial consistency can support shallow (>T) noiseless interiors without
    # pretending that a high fixed anchor threshold is a noise calibration.
    local_low = np.minimum.reduce([_shift(delta, dy, dx, np.inf)
                                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    local_high = np.maximum.reduce([_shift(delta, dy, dx, -np.inf)
                                    for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    with np.errstate(over='ignore', invalid='ignore'):
        local_range = local_high - local_low
    stable = (_erode(depth_fg, 1) & np.isfinite(local_range) &
              (local_range <= p['depth_support_tolerance_m']))
    body = (_erode(coarse, 2) & strong) | stable
    # Where RGB agrees with the empty scene, keep positive anchors farther from
    # mixed depth edges. This is not a veto of same-color objects.
    body &= ~reference_agrees | _erode(coarse, 3)
    # Opposite supported neighbors identify a thin run, including diagonals.
    # Endpoints are accepted only next to such a run, not arbitrary speckles.
    thin_run = np.zeros(valid.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        across = (_shift(valid & (d < 1), -dx, dy, False) &
                  _shift(valid & (d < 1), dx, -dy, False))
        with np.errstate(over='ignore', invalid='ignore'):
            mismatch = np.maximum(np.abs(delta - _shift(delta, dy, dx, np.nan)),
                                  np.abs(delta - _shift(delta, -dy, -dx, np.nan)))
            tolerance = p['depth_support_tolerance_m']
        cross_color = (edge_valid & _shift(edge_valid, -dx, dy, False) &
                       _shift(edge_valid, dx, -dy, False) &
                       (np.linalg.norm(lab - _shift(lab, -dx, dy), axis=2) > sigma) &
                       (np.linalg.norm(lab - _shift(lab, dx, -dy), axis=2) > sigma))
        consistent = (mismatch <= tolerance) & (cross_color | (mismatch <= .1 * tolerance))
        along = ((strong & _shift(strong, dy, dx, False) & _shift(strong, -dy, -dx, False)) |
                 (_shift(depth_fg, dy, dx, False) & _shift(depth_fg, -dy, -dx, False) & consistent))
        thin_run |= along & across
        # Aggregate along a visible narrow RGB run instead of across it. This
        # supports noisy 1px branches without a square-core requirement. Only
        # real above-threshold center depth can become a thin anchor.
        samples = []
        for step in range(-2, 3):
            neighbor_valid = _shift(valid & edge_valid, step * dy, step * dx, False)
            color_close = np.linalg.norm(lab - _shift(lab, step * dy, step * dx), axis=2) <= sigma
            samples.append(np.where(neighbor_valid & color_close,
                                    _shift(d, step * dy, step * dx, np.nan), np.nan))
        sample_stack = np.stack(samples)
        enough = np.isfinite(sample_stack).sum(axis=0) >= 3
        sample_stack[:, ~enough] = 0.
        run_depth = np.nanmedian(sample_stack, axis=0)
        thin_run |= cross_color & ~reference_agrees & enough & (run_depth > 1.)
    thin_run &= depth_fg & (strong_neighbors <= 4)
    endpoints = np.zeros(valid.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                with np.errstate(over='ignore', invalid='ignore'):
                    agrees = np.abs(delta - _shift(delta, dy, dx, np.nan)) <= .1 * tolerance
                endpoints |= _shift(thin_run, dy, dx, False) & (strong | agrees)
    thin = thin_run | (depth_fg & (strong_neighbors <= 2) & endpoints)
    coarse |= thin
    # Small coherent components need not contain an eroded 2D core either.
    from .debug import _opencv
    cv = _opencv()
    n, small_labels, stats, _ = cv.connectedComponentsWithStats(depth_fg.astype(np.uint8), connectivity=8)
    area = stats[:, cv.CC_STAT_AREA]
    count = np.bincount(small_labels[strong & depth_fg], minlength=n)
    small_keep = (area >= 3) & (area <= 25) & (count >= .8 * area)
    small_keep[0] = False
    small = small_keep[small_labels] & strong
    coarse |= small
    anchors = body | thin | small
    # A component without any reliable positive support cannot create an object
    # merely because RGB is dark/colored. Avoid an extra external dependency.
    _, labels = cv.connectedComponents(coarse.astype(np.uint8), connectivity=8)
    keep = np.zeros(int(labels.max()) + 1, bool)
    keep[labels[anchors]] = True
    keep[0] = False
    domain = keep[labels]
    for _ in range(p['anchored_refine_radius_px']):
        domain |= _neighbors8(domain) & valid & ~farther
    domain &= valid & ~farther
    raw_near_table = valid & (np.abs(d) < .5)
    near_table = raw_near_table & (np.abs(robust) < .5)
    # Preserve isolated observed table pixels (including 1px holes) when RGB
    # agrees with the reference. Without robust support they stay unknown,
    # not confident background. A median must not erase this raw counterevidence.
    raw_background_guard = raw_near_table & reference_agrees & ~near_table
    background_anchors = (_erode(near_table, 1) |
                          (near_table & reference_agrees) | farther) & ~anchors
    excluded = ~domain & valid
    negative = (excluded | background_anchors | raw_background_guard) & ~anchors
    # Internal current-image edges are useful even when subtraction was rejected
    # or the object is black/white. Missing current color gives neutral affinity.
    weights = []
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        both = valid & _shift(valid, dy, dx, False)
        color_pair = edge_valid & _shift(edge_valid, dy, dx, False)
        distance2 = np.sum((lab - _shift(lab, dy, dx)) ** 2, axis=2)
        # Divide in stages rather than square a potentially huge finite sigma.
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            scaled = distance2 / sigma / sigma
            affinity = np.where(color_pair, np.exp(-.5 * scaled), 1.)
        weights.append((dy, dx, p['anchored_pairwise_weight'] * affinity * both))
    # Bounded Jacobi relaxation of a color-weighted quadratic energy. This is
    # a deterministic continuous support score, not a posterior probability.
    unary = np.clip((robust - 1.) / 3. + .5, 0., 1.)
    score = (coarse & domain & ~negative).astype(float)
    score[anchors & domain] = 1.
    mutable = domain & ~anchors & ~negative
    data_weight = .2
    denominator = data_weight + sum(weight for _, _, weight in weights)
    iterations = 0
    for iteration in range(p['anchored_iterations']):
        numerator = data_weight * unary
        for dy, dx, weight in weights:
            numerator = numerator + weight * _shift(score, dy, dx, 0.)
        update = np.clip(numerator / denominator, 0., 1.)
        change = float(np.max(np.abs(update[mutable] - score[mutable]))) if mutable.any() else 0.
        score[mutable] = update[mutable]
        iterations = iteration + 1
        if change < 1e-5:
            break
    selected = (score > .5) & domain & ~negative
    selected |= anchors & domain
    inferred = selected & ~anchors
    # Confidence is only a conservative rule category: require independent
    # usable color subtraction and depth support away from the final boundary.
    # Same-color depth anchors and all RGB-refined geometry stay uncertain.
    confident_fg = (body & selected & _erode(selected, 1) & reliable_reference & color_fg &
                    (delta > t + p['depth_margin_m']))
    confident_bg = negative & near_table & reference_agrees
    unknown = roi & ~(confident_fg | confident_bg)
    suppressed = depth_fg & ~selected
    deferred = roi & ~selected & ~suppressed & ~confident_bg & (domain | color_fg | farther)
    decision = np.zeros(valid.shape, np.uint8)
    for code, mask in ((3, background_anchors), (4, suppressed), (5, deferred),
                       (1, selected & (body | small)), (2, selected & thin & ~body & ~small), (7, inferred)):
        decision[mask] = code
    debug = dict(fusion_decision=decision, robust_depth_ratio=robust,
                 current_color_edge_valid_mask=edge_valid,
                 foreground_anchor_mask=anchors, thin_depth_anchor_mask=thin,
                 small_component_anchor_mask=small,
                 background_anchor_mask=background_anchors, raw_background_guard_mask=raw_background_guard,
                 refinement_domain_mask=domain,
                 refinement_excluded_mask=excluded,
                 inferred_foreground_mask=inferred,
                 confident_foreground_mask=confident_fg,
                 confident_background_mask=confident_bg, unknown_mask=unknown,
                 refinement_iterations=iterations, refinement_score=score,
                 refinement_converged=change < 1e-5, refinement_max_update=change)
    return selected, suppressed, deferred, debug
