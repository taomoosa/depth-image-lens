"""Fixed seed-region color models with bounded, background-competitive inference."""
import numpy as np
from ._rgbd_color_local import color_local_fusion
from ._rgbd_fusion import _neighbors8
from .debug import _opencv

MODEL_DECISION_REASONS = {
    0: 'no_color_candidate_or_unavailable',
    1: 'depth_supported_model_training_pixel',
    3: 'color_rejected_or_disconnected_from_model',
    4: 'raw_depth_outside_model_selection',
    5: 'model_unavailable_or_depth_unavailable',
    6: 'removed_by_min_object_area',
    7: 'bounded_region_color_model_inference',
}


def _palette(colors, max_colors, tolerance, min_fraction):
    """Deterministic dense color clusters; never update from inferred pixels.

    Quantized bins locate modes without tying every unique noisy color. Each
    prototype is a coordinate-wise median of its bounded training cluster.
    Small unsupported color tails do not create independent prototypes.
    """
    remaining = np.ones(len(colors), bool)
    result = []
    minimum = max(3, int(np.ceil(len(colors) * min_fraction)))
    for _ in range(max_colors):
        indices = np.flatnonzero(remaining)
        if len(indices) < minimum:
            break
        keys, inverse, counts = np.unique(np.floor(colors[indices] / 6).astype(np.int16),
                                          axis=0, return_inverse=True, return_counts=True)
        mode = int(np.argmax(counts))
        center = np.median(colors[indices[inverse == mode]], axis=0)
        near = remaining & (np.linalg.norm(colors-center, axis=1) <= tolerance * .5)
        if near.sum() >= minimum:
            result.append(np.median(colors[near], axis=0))
        remaining[near] = False
    return result


def color_model_fusion(delta, depth_fg, farther, color_fg, color_usable,
                       color_uncertain, lab, edge_valid, color_debug, valid,
                       roi, parameters):
    p = parameters
    _, _, _, local = color_local_fusion(
        delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
        lab, edge_valid, color_debug, valid, roi, p)
    usable = color_usable & edge_valid & ~color_debug['background_photometric_clipped_mask']
    domain = local['local_candidate_mask'] & valid & ~farther & ~local['local_upper_excluded_mask']
    depth_training = domain & (delta > p['depth_threshold_m'] + p['depth_margin_m'])
    # Same-position background distance, in the same encoded full-Lab units as
    # model distance. A large change alone cannot authorize a pixel.
    background_distance = np.hypot(color_debug['background_chroma_distance'],
                                   color_debug['background_luma_distance'])
    cv = _opencv()
    count, labels, stats, _ = cv.connectedComponentsWithStats(local['local_seed_mask'].astype(np.uint8), connectivity=8)
    order = sorted(range(1,count), key=lambda k: (-int(stats[k,cv.CC_STAT_AREA]),k))
    selected = np.zeros(valid.shape,bool); trained = selected.copy(); source_mask=selected.copy()
    rejected = selected.copy(); rescued = selected.copy()
    steps = np.full(valid.shape,-1,np.int8); model_labels=np.zeros(valid.shape,np.int32)
    distances=np.full(valid.shape,np.nan); models=[]
    unavailable = np.isin(labels, order[p['model_max_regions']:])
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
        palette=_palette(lab[training],p['model_max_colors'],p['model_color_tolerance'],p['model_min_color_fraction'])
        record['colors_lab']=[c.tolist() for c in palette]
        if not palette:
            unavailable |= core
            record['status']='no_supported_color_mode';models.append(record);continue
        usable_model_count += 1
        distance=np.full(valid.shape,np.inf)
        for color in palette:
            distance=np.minimum(distance,np.linalg.norm(lab-color,axis=2))
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
    local.update(fusion_decision=decision,model_training_mask=trained,model_source_mask=source_mask,
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
    return selected,suppressed,deferred,local
