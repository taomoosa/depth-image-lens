"""Color-first candidates with conservative region-versus-surround depth evidence.

Block statistics are robustness heuristics, NOT independent samples, p-values,
or calibrated confidence. A bias confined to an object-colored region remains
indistinguishable from a shallow object in a single RGB-D observation.
"""
from math import ceil

import numpy as np
from .debug import _opencv

FIRST_DECISION_REASONS = {
    0: 'no_color_candidate', 1: 'region_supported_observed_foreground',
    3: 'region_not_separated_from_surround_depth', 4: 'raw_depth_outside_color_selection',
    5: 'insufficient_region_or_surround_evidence', 6: 'removed_by_min_object_area',
    7: 'region_supported_thin_foreground_inference',
}


def _summary(values):
    if not len(values):
        return dict(mean_m=None, median_m=None, q10_m=None, q90_m=None)
    return dict(mean_m=float(np.mean(values)), median_m=float(np.median(values)),
                q10_m=float(np.quantile(values, .1)), q90_m=float(np.quantile(values, .9)))


def _block_footprint(size):
    return max(4, size*size//2)


def _blocks(mask, valid, delta, size, coverage):
    """At least half a block's footprint, then adequate valid coverage.

    Partial boundary blocks do not count as independent strong evidence. Depth
    means include signed values, not only positive threshold crossings.
    """
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return np.empty(0)
    ids = (yy // size) * ((mask.shape[1] + size - 1)//size) + xx // size
    groups, inv, count = np.unique(ids, return_inverse=True, return_counts=True)
    seen = valid[yy, xx]
    n = np.bincount(inv, weights=seen, minlength=len(groups))
    sums = np.bincount(inv, weights=np.where(seen, delta[yy, xx], 0.), minlength=len(groups))
    ok = (count >= _block_footprint(size)) & (n >= count*coverage)
    return sums[ok] / n[ok]


def color_first_fusion(delta, depth_fg, farther, color_fg, color_usable,
                       color_uncertain, lab, edge_valid, color_debug, valid,
                       roi, parameters, *, diagnostics=False, table=None):
    p = parameters; cv = _opencv(); shape = valid.shape
    surround_override = p['first_min_surround_blocks']
    min_surround_blocks = p['first_min_blocks'] if surround_override is None else surround_override
    block_size = p['first_block_size_px']
    min_coverage = p['first_min_valid_fraction']
    min_pixel_reference = max(4, ceil(_block_footprint(block_size)*min_coverage))
    usable = roi & color_usable & edge_valid & ~color_debug['background_photometric_clipped_mask']
    color_candidate = usable & (color_debug['background_combined_score'] > p['first_color_relaxation_ratio']**2)
    candidate = color_candidate
    upper = np.zeros(shape, bool)
    if p['local_max_reference_separation_m'] is not None:
        upper = valid & (delta > p['local_max_reference_separation_m'])
    # Candidate definition is independent of depth. Upper/invalid pixels must
    # not silently disappear from the statistical coverage denominator.
    sample_valid = valid & ~upper
    table_accepted = table is not None and table[-1]['status'] == 'accepted'
    table_forced = None
    if table is not None:
        table_proposal, table_domain, table_reference, table_forced, table_report = table
        table_forced = table_forced & sample_valid & ~depth_fg & ~farther
        candidate = color_candidate & ~table_forced
        if table_accepted:
            table_report['forced_background_pixels'] = int(table_forced.sum())
            reference_floor = table_report['sample_median_m'] - table_report['offset_m']
            # Measurable positive separation is not a trustworthy table
            # reference, even if color connected it to the table component.
            # Keep lower-side noise and the original coverage denominator.
            elevated = delta > reference_floor + .5*p['first_min_depth_delta_m']
            table_reference = table_reference & sample_valid & ~elevated
            table_report['reference_pixels'] = int(table_reference.sum())
    clipped = np.clip(np.where(valid, delta, 0.), -p['local_depth_clip_m'], p['local_depth_clip_m'])
    scales = np.array([1/p['model_luma_tolerance'], 1/p['model_chroma_tolerance'], 1/p['model_chroma_tolerance']])
    color_ids = np.zeros(shape, np.int32)
    if candidate.any():
        colors, inverse, counts = np.unique(lab[candidate], axis=0, return_inverse=True, return_counts=True)
        assignment = np.zeros(len(colors), np.int32)
        # Fixed prototypes avoid transitive drift across a smooth shadow.
        for group in range(1, p['region_max_color_groups']+1):
            remaining = assignment == 0
            if not remaining.any(): break
            seed = int(np.argmax(np.where(remaining, counts, 0)))
            assignment[remaining & (np.linalg.norm((colors-colors[seed])*scales, axis=1) <= 1)] = group
        color_ids[candidate] = assignment[inverse]
    selected = np.zeros(shape, bool); interiors = selected.copy(); rings = selected.copy()
    labels_out = np.zeros(shape, np.int32); records = []
    insufficient = candidate & (color_ids == 0)
    failure_masks = {name: np.zeros(shape, bool) for name in ('area','interior','region_coverage','surround_coverage','region_blocks','surround_blocks','mean_delta','median_delta','block_fraction')}
    if surround_override is not None:
        failure_masks['surround_reference'] = np.zeros(shape, bool)
    mean_map = np.full(shape, np.nan); median_map = mean_map.copy(); threshold_map = mean_map.copy()
    next_id = 0
    for group in range(1, int(color_ids.max())+1):
        count, labels, boxes, _ = cv.connectedComponentsWithStats((color_ids == group).astype(np.uint8), connectivity=8)
        for label in range(1, count):
            next_id += 1
            area = int(boxes[label, cv.CC_STAT_AREA])
            x,y,w,h = boxes[label,:4]
            sl = np.s_[y:y+h,x:x+w]
            piece = labels[sl] == label
            labels_out[sl][piece] = next_id
            if area < p['first_min_area']:
                failure_masks['area'][sl][piece] = True; insufficient[sl][piece] = True
                records.append(dict(id=next_id,color_group=group,area=area,accepted=False,failures=['area']))
                continue
            part = labels == label
            inset = p['model_training_inset_px']
            interior = cv.erode(part.astype(np.uint8), np.ones((2*inset+1,2*inset+1),np.uint8), borderType=cv.BORDER_CONSTANT, borderValue=1).astype(bool)
            # Exclude ALL changed colors from the reference ring, not just this
            # group. Nearby objects/shadows cannot serve as trusted table.
            radius = p['first_ring_radius_px']; gap = p['first_ring_gap_px']
            ring = cv.dilate(part.astype(np.uint8), np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
            near = cv.dilate(part.astype(np.uint8), np.ones((2*gap+1,2*gap+1),np.uint8)).astype(bool)
            radius_region = ring
            ring = radius_region & ~near & usable & ~candidate
            ring_valid = sample_valid
            table_source = 'ordinary_ring'
            chosen_radius = radius
            cached_blocks = None
            if table_accepted:
                reference_valid = sample_valid & table_reference
                # Evaluate the measured local table first, then a limited wider
                # ring only if calibration tiles agree within the depth target.
                expand = min(128, 3*radius)
                radii = (radius, expand) if (
                    expand > radius and table_report['max_tile_deviation_m'] <= p['first_min_depth_delta_m']) else (radius,)
                for rr in radii:
                    outer = radius_region if rr == radius else cv.dilate(
                        part.astype(np.uint8), np.ones((2*rr+1,2*rr+1),np.uint8)).astype(bool)
                    probe = outer & ~near & table_domain
                    num = int((probe & reference_valid).sum())
                    if num < min_coverage*max(1, int(probe.sum())):
                        continue
                    blocks = _blocks(probe, reference_valid, clipped, block_size, min_coverage)
                    if len(blocks) < min_surround_blocks or (len(blocks) < 2 and num < min_pixel_reference):
                        continue
                    ring, ring_valid, cached_blocks = probe, reference_valid, blocks
                    table_source = 'anchor_local_ring' if rr == radius else 'anchor_expanded_ring'
                    chosen_radius = rr
                    break
            interiors |= interior; rings |= ring
            iv = interior & sample_valid; rv = ring & ring_valid
            ring_values = clipped[rv]
            ic = float(iv.sum()/max(1,interior.sum())); rc = float(len(ring_values)/max(1,ring.sum()))
            ib = _blocks(interior,sample_valid,clipped,block_size,min_coverage)
            rb = _blocks(ring,ring_valid,clipped,block_size,min_coverage) if cached_blocks is None else cached_blocks
            region = _summary(clipped[iv]); surround = _summary(ring_values)
            gates = dict(area=True,interior=bool(interior.any()),region_coverage=ic>=min_coverage,surround_coverage=rc>=min_coverage,region_blocks=len(ib)>=p['first_min_blocks'],surround_blocks=len(rb)>=min_surround_blocks)
            use_pixels = surround_override is not None and len(rb) < 2 and gates['surround_blocks']
            if use_pixels:
                # One block mean gives an artificial zero spread, and zero
                # block means give no estimator. Use real ring observations
                # only, with a sample floor even at 100% ring coverage.
                gates['surround_reference'] = len(ring_values) >= min_pixel_reference
            enough = all(gates.values())
            mean_delta = median_delta = required = fraction = None
            reference_source = noise = None
            if enough:
                mean_base, median_base = surround['mean_m'], surround['median_m']
                if table_source != 'ordinary_ring':
                    # Rejecting strong positive candidates must not lower the
                    # table baseline below the independently fitted level.
                    mean_base = max(mean_base, reference_floor)
                    median_base = max(median_base, reference_floor)
                mean_delta = region['mean_m']-mean_base
                median_delta = region['median_m']-median_base
                reference_values = ring_values if use_pixels else rb
                reference_median = np.median(reference_values)
                reference_source = 'ring_pixels' if use_pixels else 'block_means'
                # Use the observed spread without dividing by sqrt(N).
                # Large images do not turn correlated table bias into certainty.
                noise = float(np.quantile(np.abs(reference_values-reference_median), .9))
                if table_source != 'ordinary_ring':
                    noise = max(noise, table_report['tile_p90_deviation_m'])
                    reference_median = max(reference_median, reference_floor)
                required = p['first_min_depth_delta_m'] + p['first_noise_multiplier']*noise
                fraction = float(np.mean(ib-reference_median >= required))
                gates.update(mean_delta=mean_delta>=required,median_delta=median_delta>=required,block_fraction=fraction>=p['first_min_block_fraction'])
                mean_map[part]=mean_delta; median_map[part]=median_delta; threshold_map[part]=required
            else:
                insufficient |= part
            accepted = enough and all(gates.values())
            if accepted: selected |= part & sample_valid
            for name, passed in gates.items():
                if not passed: failure_masks[name] |= part
            record = dict(id=next_id,color_group=group,area=area,interior_area=int(interior.sum()),surround_area=int(ring.sum()),region_valid_fraction=ic,surround_valid_fraction=rc,region_blocks=len(ib),surround_blocks=len(rb),region=region,surround=surround,mean_delta_m=mean_delta,median_delta_m=median_delta,required_delta_m=required,supported_block_fraction=fraction,accepted=accepted,failures=[k for k,v in gates.items() if not v])
            if surround_override is not None:
                record.update(surround_min_blocks=min_surround_blocks,
                              surround_valid_pixels=len(ring_values),
                              surround_min_reference_pixels=min_pixel_reference if use_pixels else None,
                              surround_reference_source=reference_source,
                              surround_noise_m=noise)
            if table is not None:
                record.update(table_reference_source=table_source,
                              table_reference_radius_px=chosen_radius)
            records.append(record)
    inferred = selected & ~depth_fg
    suppressed = depth_fg & ~selected
    deferred = candidate & ~selected & ~suppressed
    decision = np.zeros(shape,np.uint8); decision[deferred]=3; decision[suppressed]=4; decision[insufficient | (candidate & ~sample_valid)]=5; decision[selected & depth_fg]=1; decision[inferred]=7
    unknown = roi & (~valid | ~usable | inferred | suppressed | deferred | upper)
    debug = dict(fusion_decision=decision,first_candidate_mask=color_candidate,first_color_group_map=color_ids,first_region_label_map=labels_out,first_interior_mask=interiors,first_surround_mask=rings,first_upper_excluded_mask=upper,first_insufficient_mask=insufficient,first_mean_delta_m=mean_map,first_median_delta_m=median_map,first_required_delta_m=threshold_map,first_statistics=records,first_color_budget_exceeded=bool((candidate & (color_ids==0)).any()),accepted_both_mask=selected & depth_fg,accepted_depth_only_mask=np.zeros(shape,bool),accepted_color_only_mask=np.zeros(shape,bool),accepted_growth_mask=inferred,inferred_foreground_mask=inferred,confident_foreground_mask=selected & depth_fg & ~color_uncertain,confident_background_mask=np.zeros(shape,bool),unknown_mask=unknown)
    if table_forced is not None:
        decision[table_forced] = 8
        unknown &= ~table_forced
        debug['confident_background_mask'] = table_forced.copy()
        debug['first_table_forced_background_mask'] = table_forced
        if diagnostics:
            debug.update(first_table_proposal_mask=table_proposal,
                         first_table_reference_domain_mask=table_domain,
                         first_table_reference_mask=table_reference)
    if diagnostics:
        debug.update({f'first_failed_{key}_mask': value for key,value in failure_masks.items()})
    return selected,suppressed,deferred,debug
