"""In-memory RGB-D assistance; synthetic example, no file or camera required."""
from dataclasses import replace
import json
import numpy as np
from depth_image_lens import BridgeDrapeDetector, SeededColorAssistant, analyze_color_geometry
from depth_image_lens.depth import depth_pair_16uc1_to_meters


def apply_color_constraints(detector, depth, difference, hand_rois, baseline, colors, *,
                            depth_scale_m=.001, local_anchor_support=False, semantic_path_policy="component", **depth_options):
    """Experimental policy: depth connectivity plus conditional hand exclusion.

    Uncertain color is not missing depth. Both separated appearance interiors
    are required before applying the RGB exclusion. Otherwise the RGB result
    abstains; the independently returned depth-only observation is not replaced.
    """
    if semantic_path_policy not in ("component", "observed_route"):
        raise ValueError("semantic_path_policy must be component or observed_route")
    if semantic_path_policy == "observed_route" and detector.max_gap_px:
        raise ValueError("observed_route requires max_gap_px=0")
    current,_=depth_pair_16uc1_to_meters(depth,difference,depth_scale_m=depth_scale_m)
    geometry_options={k:v for k,v in depth_options.items() if k in
        ('observed_mask','arm_mask','work_roi','hand_visibility','down','depth_jump_m')}
    if depth_options.get('background_valid') is not None:
        prior=geometry_options.get('observed_mask')
        geometry_options['observed_mask']=depth_options['background_valid'] if prior is None else prior&depth_options['background_valid']
    color_geometry=analyze_color_geometry(detector,current,colors,hand_rois,depth_unit='m',
        depth_geometry=baseline,diagnostics=True,**geometry_options)
    if hand_rois is None:return color_geometry
    route = None
    depth_candidates = None
    if baseline.foreground_mask.any() and not (colors.object_mask.any() and colors.hand_mask.any()):
        diagnostics = dict(baseline.diagnostics)
        diagnostics['rgb_geometry_sag_path_used'] = diagnostics.get('sag_path_used',False)
        diagnostics['sag_path_used'] = False
        result = replace(baseline, bridge='unknown', large_sag='unknown', drape='unknown',
                         diagnostics=diagnostics,
                         reasons=dict.fromkeys(('bridge','large_sag','drape'),'rgb_classes_not_separable'))
    else:
        anchors = baseline.diagnostics['left_anchor_mask'] | baseline.diagnostics['right_anchor_mask']
        exclusion = colors.hand_exclusion_mask & ~anchors
        options = dict(depth_options)
        if local_anchor_support: options["anchor_policy"] = "local_observed"
        if options.get('arm_mask') is not None:
            exclusion = exclusion | options.pop('arm_mask')
        else:
            options.pop('arm_mask', None)
        result = detector.detect_16uc1(depth, difference, hand_rois, depth_scale_m=depth_scale_m,
                                      arm_mask=exclusion, diagnostics=True, **options)
        depth_candidates = {'bridge':result.bridge, 'large_sag':result.large_sag, 'drape':result.drape}
        # Keep depth connectivity observable, but do not call an unclassified
        # appearance path an object bridge. Guard-only pixels are not this mask.
        unresolved = colors.appearance_uncertain_mask & ~anchors
        reasons = dict(result.reasons)
        bridge, sag, drape = result.bridge, result.large_sag, result.drape
        if bridge == 'yes' and np.any(unresolved & result.diagnostics['bridge_corridor_mask']):
            if semantic_path_policy == "observed_route":
                # Unknown appearance is excluded only from this positive route
                # proof. It is not relabeled as sensor-invalid or observed empty.
                candidate = detector.detect_16uc1(depth, difference, hand_rois, depth_scale_m=depth_scale_m,
                    arm_mask=exclusion | unresolved, diagnostics=True, **options)
                if candidate.bridge == 'yes': route = candidate
            if route is None:
                bridge = sag = 'unknown'
                reasons['bridge'] = reasons['large_sag'] = 'rgb_unresolved_appearance_on_depth_path'
            else:
                sag = route.large_sag
                reasons['bridge'] = 'rgb_observed_classified_route_reaches_both_anchors'
                reasons['large_sag'] = route.reasons['large_sag']
        if drape != 'unknown' and any(np.any(unresolved & result.diagnostics[name+'_drape_component_mask'])
                                     for name in ('left','right')):
            drape = 'unknown'; reasons['drape'] = 'rgb_unresolved_appearance_on_drape_component'
        diagnostics = dict(result.diagnostics)
        if route is not None:
            diagnostics['rgb_component_before_route_mask'] = diagnostics['bridge_corridor_mask']
            diagnostics['bridge_corridor_mask'] = route.diagnostics['bridge_corridor_mask']
            diagnostics['rgb_semantic_route_exclusion_mask'] = unresolved
            diagnostics['sag_path_used'] = route.diagnostics['sag_path_used']
            if 'sag_path_xy' in route.diagnostics: diagnostics['sag_path_xy'] = route.diagnostics['sag_path_xy']
            result = replace(result, features={**result.features,
                **{key: route.features[key] for key in ('sag_ratio', 'span_coverage')},
                'rgb_verified_route_geometry': route.features})
        diagnostics['rgb_geometry_sag_path_used'] = diagnostics.get('sag_path_used',False)
        if sag == 'unknown': diagnostics['sag_path_used'] = False
        result = replace(result, bridge=bridge, large_sag=sag, drape=drape, reasons=reasons, diagnostics=diagnostics)
    features = {**result.features, 'rgb_integration_policy':'depth_connectivity_with_color_exclusion',
                'rgb_semantic_uncertain_pixels':colors.summary['appearance_uncertain_pixels'],
                'rgb_guard_only_uncertain_pixels':colors.summary['uncertain_pixels']-colors.summary['appearance_uncertain_pixels'],
                'rgb_connectivity_scope':'valid depth foreground; unresolved appearance may be present',
                'rgb_physical_contact_verified':False, 'rgb_local_anchor_support_enabled':local_anchor_support, 'rgb_depth_geometry_before_semantic_gate':depth_candidates}
    if semantic_path_policy != "component":
        features.update(rgb_semantic_path_policy=semantic_path_policy, rgb_semantic_route_used=route is not None)
    stages={**result.stage_results,'semantic_classification':{
                'status':'computed','bridge':result.bridge,'large_sag':result.large_sag,'drape':result.drape,
                'reasons':result.reasons,'physical_contact_verified':False},
            'appearance':color_geometry.stage_results['appearance'],
            'color_candidate_geometry':color_geometry.stage_results,
            'depth_candidate_geometry':baseline.stage_results}
    return replace(result, features=features,stage_results=stages)


def run(depth, difference, color, hand_rois, object_seeds, hand_seeds, *,
        color_order, registered_to_depth, depth_scale_m=.001, boundary_guard_px=1, local_anchor_support=False, semantic_path_policy="component",
        observed_mask=None, background_valid=None, arm_mask=None, color_valid_mask=None):
    detector = BridgeDrapeDetector()
    depth_options = dict(observed_mask=observed_mask, background_valid=background_valid, arm_mask=arm_mask)
    baseline = detector.detect_16uc1(depth, difference, hand_rois,
                                    depth_scale_m=depth_scale_m, diagnostics=True, **depth_options)
    colors = SeededColorAssistant(boundary_guard_px=boundary_guard_px).segment_16uc1(
        depth, difference, color, object_seeds=object_seeds, hand_seeds=hand_seeds,
        color_order=color_order, registered_to_depth=registered_to_depth,
        depth_scale_m=depth_scale_m, threshold_m=detector.threshold_m,
        observed_mask=observed_mask, background_valid=background_valid, color_valid_mask=color_valid_mask)
    assisted = apply_color_constraints(detector, depth, difference, hand_rois, baseline, colors,
                                        depth_scale_m=depth_scale_m, local_anchor_support=local_anchor_support, semantic_path_policy=semantic_path_policy, **depth_options)
    return baseline, colors, assisted


def main():
    shape = (80, 120)
    obj = np.zeros(shape, bool); obj[20:27, 20:101] = True
    hands = np.zeros(shape, bool); hands[10:18, 17:24] = True; hands[10:18, 97:104] = True
    depth = np.where(obj | hands, 1500, 2000).astype(np.uint16)
    difference = np.where(obj | hands, 500, 0).astype(np.uint16)
    color = np.full((*shape, 3), 40, np.uint8)
    color[obj] = (30, 70, 200); color[hands] = (160, 130, 100)
    object_seeds = np.zeros(shape, bool); object_seeds[23, 60] = True
    hand_seeds = np.zeros(shape, bool); hand_seeds[13, 20] = True
    boxes = {'left': (14, 4, 26, 20), 'right': (94, 4, 106, 20)}
    baseline, colors, assisted = run(depth, difference, color, boxes, object_seeds, hand_seeds,
                                    color_order='RGB', registered_to_depth=True)
    print(json.dumps({'depth_only':baseline.to_dict(), 'color_candidates':colors.to_dict(),
                      'rgb_assisted':assisted.to_dict()}, indent=2, allow_nan=False))


if __name__ == '__main__': main()
