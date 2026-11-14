"""Fixed seed-region color models with bounded, background-competitive inference."""
import textwrap
import numpy as np
from ._rgbd_color_local import color_local_fusion
from ._rgbd_fusion import _neighbors8
from .debug import _opencv

MODEL_REJECTION_FLAGS = {
    1: 'outside_roi', 2: 'color_unusable_or_clipped', 4: 'depth_invalid',
    8: 'not_local_color_candidate', 16: 'farther_than_background',
    32: 'upper_depth_excluded', 64: 'model_color_distance_exceeded',
    128: 'background_competition_failed', 256: 'model_has_no_color_qualified_source',
    512: 'color_qualified_but_disconnected',
    1024: 'connected_but_growth_budget_exceeded',
}

MODEL_DECISION_REASONS = {
    0: 'no_color_candidate_or_unavailable',
    1: 'depth_supported_model_training_pixel',
    3: 'color_rejected_or_disconnected_from_model',
    4: 'raw_depth_outside_model_selection',
    5: 'model_unavailable_or_depth_unavailable',
    6: 'removed_by_min_object_area',
    7: 'bounded_region_color_model_inference',
}


def _palette(colors, max_colors, tolerance, min_fraction, scales=None):
    """Deterministic dense color clusters; never update from inferred pixels.

    Quantized bins locate modes without tying every unique noisy color. Each
    prototype is a coordinate-wise median of its bounded training cluster.
    Small unsupported color tails do not create independent prototypes.
    """
    scales = np.ones(3) if scales is None else np.asarray(scales)
    remaining = np.ones(len(colors), bool)
    result = []
    minimum = max(3, int(np.ceil(len(colors) * min_fraction)))
    for _ in range(max_colors):
        indices = np.flatnonzero(remaining)
        if len(indices) < minimum:
            break
        keys, inverse, counts = np.unique(np.floor(colors[indices] * scales / 6).astype(np.int64),
                                          axis=0, return_inverse=True, return_counts=True)
        mode = int(np.argmax(counts))
        center = np.median(colors[indices[inverse == mode]], axis=0)
        near = remaining & (np.linalg.norm((colors-center) * scales, axis=1) <= tolerance * .5)
        if near.sum() >= minimum:
            result.append(np.median(colors[near], axis=0))
        remaining[near] = False
    return result


def color_model_fusion(delta, depth_fg, farther, color_fg, color_usable,
                       color_uncertain, lab, edge_valid, color_debug, valid,
                       roi, parameters, diagnostics=False):
    p = parameters
    scales = np.array([p['model_color_tolerance']/p['model_luma_tolerance'],
                       p['model_color_tolerance']/p['model_chroma_tolerance'],
                       p['model_color_tolerance']/p['model_chroma_tolerance']])
    raw_farther = farther
    farther_threshold = p['depth_threshold_m'] if p['model_farther_threshold_m'] is None else p['model_farther_threshold_m']
    farther = (valid & (delta < -farther_threshold) if p['model_reject_farther']
               else np.zeros(valid.shape, bool))
    _, _, _, local = color_local_fusion(
        delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
        lab, edge_valid, color_debug, valid, roi, p)
    usable = color_usable & edge_valid & ~color_debug['background_photometric_clipped_mask']
    domain = local['local_candidate_mask'] & valid & ~farther & ~local['local_upper_excluded_mask']
    depth_training = domain & (delta > p['depth_threshold_m'] + p['depth_margin_m'])
    # Same-position background distance, in the same scaled Lab metric as
    # model distance. A large change alone cannot authorize a pixel.
    background_distance = np.hypot(color_debug['background_chroma_distance'] * scales[1],
                                   color_debug['background_luma_distance'] * scales[0])
    cv = _opencv()
    count, labels, stats, _ = cv.connectedComponentsWithStats(local['local_seed_mask'].astype(np.uint8), connectivity=8)
    order = sorted(range(1,count), key=lambda k: (-int(stats[k,cv.CC_STAT_AREA]),k))
    selected = np.zeros(valid.shape,bool); trained = selected.copy(); source_mask=selected.copy()
    rejected = selected.copy(); rescued = selected.copy()
    steps = np.full(valid.shape,-1,np.int8); model_labels=np.zeros(valid.shape,np.int32)
    distances=np.full(valid.shape,np.nan); models=[]
    unavailable = np.isin(labels, order[p['model_max_regions']:])
    evaluations = []
    usable_model_count = 0
    budget_exceeded = len(order) > p['model_max_regions']
    for label in order[:p['model_max_regions']]:
        core = labels == label
        training = core & depth_training
        n = int(training.sum())
        record=dict(id=label,seed_area=int(core.sum()),training_pixels=n,colors_lab=[])
        if n < p['model_min_training_pixels']:
            unavailable |= core
            record['status']='insufficient_depth_training';models.append(record);continue
        palette=_palette(lab[training],p['model_max_colors'],p['model_color_tolerance'],p['model_min_color_fraction'],scales)
        record['colors_lab']=[c.tolist() for c in palette]
        if not palette:
            unavailable |= core
            record['status']='no_supported_color_mode';models.append(record);continue
        usable_model_count += 1
        distance=np.full(valid.shape,np.inf)
        if diagnostics:
            best_luma=np.full(valid.shape,np.nan,np.float32)
            best_chroma=np.full(valid.shape,np.nan,np.float32)
            best_prototype=np.full(valid.shape,-1,np.int16)
        for palette_index,color in enumerate(palette):
            difference=lab-color
            candidate_distance=np.linalg.norm(difference * scales,axis=2)
            if diagnostics:
                better=candidate_distance < distance
                best_luma[better]=np.abs(difference[...,0][better])
                best_chroma[better]=np.hypot(difference[...,1],difference[...,2])[better]
                best_prototype[better]=palette_index
            distance=np.minimum(distance,candidate_distance)
        color_ok=(domain & (distance <= p['model_color_tolerance']) &
                  (distance + p['model_background_margin'] < background_distance))
        source=core & color_ok
        # Reachability is recomputed in the fixed color domain, not from a
        # changing per-frontier prototype. Table-like pixels may be inferred
        # only inside this explicitly bounded boundary band.
        connected=source.copy();frontier=source.copy()
        local_steps=np.full(valid.shape,-1,np.int8);local_steps[source]=0
        for step in range(1,p['model_boundary_radius_px']+1):
            frontier=_neighbors8(frontier) & color_ok & ~connected
            if not frontier.any():break
            connected|=frontier;local_steps[frontier]=step
        if diagnostics:
            # Independent tests are retained per model: no winner-takes-all
            # explanation and no accidental combination of different models.
            flags = np.zeros(valid.shape, np.uint16)
            flags[~roi] |= 1
            flags[~usable] |= 2
            flags[~valid] |= 4
            flags[~local['local_candidate_mask']] |= 8
            flags[farther] |= 16
            flags[local['local_upper_excluded_mask']] |= 32
            flags[usable & (distance > p['model_color_tolerance'])] |= 64
            flags[usable & ~(distance + p['model_background_margin'] < background_distance)] |= 128
            source_distance = np.full(valid.shape, np.nan, np.float32)
            reachable = np.zeros(valid.shape, bool)
            if source.any():
                # Exact obstacle-free 8-neighbor distance, NOT geodesic distance.
                source_distance = cv.distanceTransform((~source).astype(np.uint8), cv.DIST_C, 3)
                _, components = cv.connectedComponents(color_ok.astype(np.uint8), connectivity=8)
                reachable = np.isin(components, np.unique(components[source])) & color_ok
                flags[color_ok & ~reachable] |= 512
                flags[reachable & ~connected] |= 1024
            else:
                flags |= 256
            measured_distance = distance.astype(np.float32)
            measured_distance[~usable] = np.nan
            best_luma[~usable]=np.nan; best_chroma[~usable]=np.nan; best_prototype[~usable]=-1
            evaluations.append((label, measured_distance, source_distance, flags, reachable, connected.copy(),
                                best_luma,best_chroma,best_prototype))
        new=connected & ~selected
        model_labels[new]=label;distances[new]=distance[new];steps[new]=local_steps[new]
        selected|=connected;trained|=training & connected;source_mask|=source
        rejected|=domain & core & ~color_ok
        record['status']='accepted' if connected.any() else 'background_or_color_rejected'
        record['selected_pixels']=int(connected.sum());models.append(record)
    if not usable_model_count:
        unavailable |= domain
    unavailable &= ~selected
    inferred=selected & ~trained
    rescued=selected & local['local_background_mask']
    suppressed=depth_fg & ~selected
    deferred=local['local_candidate_mask'] & ~selected & ~suppressed
    unknown=roi & (~valid | ~usable | inferred | suppressed | deferred | local['local_upper_excluded_mask'])
    decision=np.zeros(valid.shape,np.uint8)
    decision[deferred]=3;decision[suppressed]=4
    decision[unavailable | (local['local_candidate_mask'] & ~valid) | local['local_upper_excluded_mask']]=5
    decision[trained]=1;decision[inferred]=7
    local.update(model_raw_farther_mask=raw_farther.copy(),
                 model_farther_excluded_mask=farther.copy(),
                 model_accepted_raw_farther_mask=selected & raw_farther,
                 fusion_decision=decision,model_training_mask=trained,model_source_mask=source_mask,
                 model_rejected_core_mask=rejected,model_rescued_background_mask=rescued,
                 model_unavailable_mask=unavailable,
                 model_label_map=model_labels,model_color_distance=distances,
                 model_background_distance=background_distance,model_statistics=models,
                 model_budget_exceeded=budget_exceeded,growth_step=steps,
                 accepted_both_mask=trained,accepted_depth_only_mask=np.zeros(valid.shape,bool),
                 accepted_color_only_mask=np.zeros(valid.shape,bool),accepted_growth_mask=inferred.copy(),
                 inferred_foreground_mask=inferred,confident_foreground_mask=trained & ~color_uncertain,
                 confident_background_mask=local['local_background_mask'] & ~selected,
                 unknown_mask=unknown)
    if diagnostics:
        shape = (0,) + valid.shape
        local['model_diagnostic_ids'] = np.array([e[0] for e in evaluations], np.int32)
        for name, index, dtype in (
            ('model_all_color_distances', 1, np.float32),
            ('model_all_source_distances_px', 2, np.float32),
            ('model_all_rejection_flags', 3, np.uint16),
            ('model_all_reachable_masks', 4, bool),
            ('model_all_selected_masks', 5, bool),
            ('model_all_luma_distances', 6, np.float32),
            ('model_all_chroma_distances', 7, np.float32),
            ('model_all_nearest_palette_indices', 8, np.int16)):
            local[name] = (np.stack([e[index] for e in evaluations]) if evaluations
                           else np.empty(shape, dtype=dtype))
        local['model_no_evaluable_model_mask'] = roi & (not evaluations)
    return selected,suppressed,deferred,local


def render_model_rejection_maps(diagnostics, object_mask, width, *, model_id=None,
                                color_order='RGB'):
    """One spatial tile per independent test; never collapse reasons to a winner.

    The default is an existential test across evaluable models, not a global
    rejection proof. A selected model ID instead shows exactly that model.
    Temporary memory is O(HW), apart from the output image and existing stacks.
    """
    cv = _opencv()
    flags = diagnostics['model_all_rejection_flags']
    ids = diagnostics['model_diagnostic_ids']
    if model_id is not None:
        if isinstance(model_id, (bool, np.bool_)) or not isinstance(model_id, (int, np.integer)):
            raise ValueError('model_diagnostic_id must be an evaluable integer model ID')
        indices = np.flatnonzero(ids == model_id)
        if not len(indices):
            raise ValueError('model_diagnostic_id must be an evaluable integer model ID')
        model_flags = flags[int(indices[0])]
        scope = f'Model {int(model_id)} only'
    else:
        model_flags = None
        scope = 'ANY evaluable model (not ALL models)'
    cols = 3
    tile_width = max(1, width // cols)
    h,w = object_mask.shape
    tile_height = min(240, max(2, round(h * tile_width / w)))
    header = 76
    label_height = 56
    rows = (len(MODEL_REJECTION_FLAGS) + 1 + cols - 1) // cols
    canvas = np.zeros((header + rows * (tile_height + label_height), width, 3),np.uint8)
    for i,text in enumerate((scope + ': unselected pixels only',
                             'Orange: test failed. Gray: selected object context.',
                             'Separate tiles overlap; no priority or unique cause.',
                             'Inspect per-model flags / example for exact pixel values.')):
        cv.putText(canvas,text,(8,16+18*i),cv.FONT_HERSHEY_SIMPLEX,.38,(220,220,220),1)
    unselected = ~object_mask
    orange = (255,125,35) if color_order == 'RGB' else (35,125,255)
    tests = list(MODEL_REJECTION_FLAGS.items()) + [(0,'no_evaluable_model_in_ROI')]
    for index,(bit,reason) in enumerate(tests):
        if bit == 0:
            failed = diagnostics['model_no_evaluable_model_mask'] & unselected
        elif model_flags is not None:
            failed = ((model_flags & bit) != 0) & unselected
        else:
            # Iterate rather than materializing an MxHxW bitwise temporary.
            failed = np.zeros(object_mask.shape,bool)
            for one_model in flags:
                failed |= (one_model & bit) != 0
            failed &= unselected
        tile = np.zeros((h,w,3),np.uint8)
        tile[object_mask] = (70,70,70)
        tile[failed] = orange
        tile = cv.resize(tile,(tile_width,tile_height),interpolation=cv.INTER_NEAREST)
        y = header + (index // cols) * (tile_height + label_height)
        x = (index % cols) * tile_width
        for line_index, line in enumerate(textwrap.wrap(f'{bit}: {reason}', width=34)[:2]):
            cv.putText(canvas,line,(x+4,y+14+14*line_index),cv.FONT_HERSHEY_SIMPLEX,.32,(220,220,220),1)
        cv.putText(canvas,f'pixels: {int(failed.sum())}',(x+4,y+48),cv.FONT_HERSHEY_SIMPLEX,.34,(220,220,220),1)
        canvas[y+label_height:y+label_height+tile_height,x:x+tile_width] = tile
    return canvas
