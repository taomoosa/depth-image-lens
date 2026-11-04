"""Connect existing result classes to partial outline matching.

Run: PYTHONPATH=src python examples/example_outline_match_sources.py
Optional: --output-dir /existing/parent/new-folder

Every detector below runs on a synthetic scene. Matching consumes a visible
object mask or an ordered closed exterior, never a contact/wrinkle/height mask.
See OUTLINE_MATCH.md for the class-by-class input and uncertainty contracts.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from depth_image_lens import (
    BridgeDrapeDetector, SeededColorAssistant, analyze_color_geometry,
    ReferenceGripperSegmenter, RGBDBackgroundSegmenter, OverheadBoundaryDetector,
    OverheadCornerDetector, OuterContourDetector, PartialOutlineMatcher,
    OutlineArticulation, DepthSurfaceDetector, OuterEdgeBulgeDetector,
    OverheadPickCandidateDetector, LiftDetector, IndependentGraspDetector,
    ObjectOnlyGraspDetector, CameraIntrinsics, TablePlane,
)


def match_mask(target_xy, object_mask, *, observed_mask, matcher, articulation):
    """observed_mask describes silhouette availability, not depth-fit success.

    Retain the observation/exclusion domain supplied to the original detector.
    Do not use object_mask itself as observed_mask: that invalidates its border.
    """
    outline = OuterContourDetector().detect(object_mask, observed_mask=observed_mask)
    return outline.match_partial(target_xy, matcher=matcher, articulation=articulation)


def observed_exterior_edges(points_xy, observed_mask):
    """Availability for integer-pixel exteriors from Boundary/Corner results.

    Check the complete rasterized edge, including long CHAIN_APPROX_SIMPLE edges.
    The 2px support radius matches OuterContourDetector's default clip margin 1.
    This preserves declared missing evidence; it does not validate segmentation.
    For independent subpixel curves, supply their own per-edge validity instead.
    """
    points = np.asarray(points_xy)
    observed = np.asarray(observed_mask)
    if observed.dtype != bool or observed.ndim != 2 or min(observed.shape) < 2:
        raise ValueError('observed_mask must be bool HxW')
    if (points.dtype.kind not in 'iuf' or points.ndim != 2 or points.shape[1] != 2
            or len(points) < 3 or not np.isfinite(points).all()
            or not np.equal(points, np.rint(points)).all()):
        raise ValueError('at least three ordered integer-pixel XY vertices required')
    h, w = observed.shape
    bounded = ((points[:,0] >= 0) & (points[:,0] < w) &
               (points[:,1] >= 0) & (points[:,1] < h))
    safe = cv2.erode(observed.astype(np.uint8), np.ones((5,5),np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    valid = np.zeros(len(points), bool)
    for i in range(len(points)):
        j = (i+1) % len(points)
        if not (bounded[i] and bounded[j]):
            continue
        a, b = points[i].astype(int), points[j].astype(int)
        lo, hi = np.minimum(a,b), np.maximum(a,b)
        region = safe[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        edge = np.zeros(region.shape,np.uint8)
        cv2.line(edge,tuple(a-lo),tuple(b-lo),1,1,cv2.LINE_8)
        valid[i] = bool(region[edge > 0].all())
    return valid


def match_exteriors(target_xy, exteriors, *, observed_mask, matcher, articulation,
                    source_statuses=None):
    """One report per exterior, not an aggregate decision across objects.

    Unlike outline.match_partial(), separate match_contour() calls do not detect
    ambiguity BETWEEN objects. Keep the source index; do not select the first hit.
    Tiny contours are recorded as unavailable instead of silently dropped.
    """
    if source_statuses is not None and len(source_statuses) != len(exteriors):
        raise ValueError('one source status per exterior required')
    rows = []
    for index, points in enumerate(exteriors):
        if source_statuses is not None and source_statuses[index] != 'computed':
            rows.append(dict(source_index=index,result=None,
                             reason='source_status_not_computed: '+source_statuses[index]))
            continue
        if len(points) < 3:
            rows.append(dict(source_index=index, result=None, reason='fewer_than_three_vertices'))
            continue
        edges = observed_exterior_edges(points, observed_mask)
        result = matcher.match_contour(target_xy, points, observed_edges=edges,
                                       articulation=articulation)
        rows.append(dict(source_index=index, result=result, reason=None))
    return rows


def make_scene():
    target = np.array([[60,155],[60,70],[30,85],[20,60],[65,30],[90,35]],float)
    polygon = np.vstack([target, [[110,35],[135,30],[180,60],[170,85],[140,70],[140,155]]])
    mask = np.zeros((240,260),np.uint8)
    cv2.fillPoly(mask,[polygon.astype(np.int32)],1)
    obj = mask > 0
    empty_color = np.full((*obj.shape,3),140,np.uint8)
    empty_depth = np.full(obj.shape,2000,np.uint16)
    color, depth = empty_color.copy(), empty_depth.copy()
    color[obj] = (30,160,220)
    depth[obj] = 1950
    observed = np.ones(obj.shape,bool)  # Synthetic scene is completely visible.
    hands = {key: np.zeros_like(obj) for key in ('left','right')}
    hands['left'][188:208,7:14] = True
    hands['right'][188:208,226:233] = True
    return dict(target=target, polygon=polygon, object_mask=obj, observed=observed,
                color=color, depth=depth, empty_color=empty_color,
                empty_depth=empty_depth, hands=hands)


def run_examples():
    s = make_scene()
    target, obj, observed = s['target'], s['object_mask'], s['observed']
    color, depth = s['color'], s['depth']
    matcher = PartialOutlineMatcher()
    sleeve = OutlineArticulation(2,3,(62.5,50),max_angle_deg=30,angle_step_deg=10)
    options = dict(matcher=matcher, articulation=sleeve)
    reports = {}

    # 1. An existing bool mask, or a caller's ordered closed XY contour.
    reports['bool_mask'] = match_mask(target,obj,observed_mask=observed,**options)
    reports['closed_xy'] = matcher.match_contour(target,s['polygon'],articulation=sleeve)

    # 2. OuterContourResult already carries exterior identity and observation.
    outline = OuterContourDetector().detect(obj,observed_mask=observed)
    reports['OuterContourResult'] = outline.match_partial(target,**options)
    # Reuse one component, preserving the original observation domain and holes.
    reports['OuterContourResult.component_mask'] = match_mask(
        target,outline.component_mask(),observed_mask=outline.observed_mask,**options)

    # 3. BoundaryResult.object_mask: never color/depth_boundary_mask or trails.
    boundary = OverheadBoundaryDetector(trace_boundaries=True).detect(
        color,depth,color_order='RGB',registered_to_depth=True,depth_scale_m=.001,
        background_color=s['empty_color'])
    # Here the synthetic reference is aligned/valid. Inspect stages and uncertainty
    # on real data; boundary.uncertain_mask is NOT a generic observation mask.
    reports['BoundaryResult.object_mask'] = match_mask(
        target,boundary.object_mask,observed_mask=observed,**options)

    # 4. RGBDBackgroundResult: current public modes expose observed_mask.
    # Extended implementations may provide outline_observed_mask to preserve
    # suppressed/deferred decision gaps; prefer it when available.
    background = RGBDBackgroundSegmenter().detect(
        color,depth,background_color=s['empty_color'],background_depth=s['empty_depth'],
        color_order='RGB',registered_to_depth=True,depth_scale_m=.001)
    background_observed = getattr(background,'outline_observed_mask',background.observed_mask)
    reports['RGBDBackgroundResult'] = match_mask(target,background.object_mask,
        observed_mask=background_observed,**options)

    # 5. CornerResult stores a whole object mask, not only corner candidates.
    corner = OverheadCornerDetector().detect(color,depth,color_order='RGB',
        registered_to_depth=True,object_mask=obj,diagnostics=True)
    reports['CornerResult.object_mask'] = match_mask(
        target,corner.object_mask,observed_mask=observed,**options)

    # 6. ColorAssistResult: two seed classes, no artificial boundary erosion here.
    hand_union = s['hands']['left'] | s['hands']['right']
    with_hands = color.copy()
    with_hands[hand_union] = (210,65,45)
    object_seed = np.zeros_like(obj)
    object_seed[100:105,95:100] = True
    colors = SeededColorAssistant(boundary_guard_px=0).segment(
        with_hands,obj|hand_union,object_seeds=object_seed,hand_seeds=s['hands']['left'],
        color_order='RGB',registered_to_depth=True)
    color_observed = observed & ~colors.uncertain_mask & ~colors.hand_exclusion_mask
    reports['ColorAssistResult'] = match_mask(
        target,colors.object_mask,observed_mask=color_observed,**options)

    # 7. ReferenceSegmentationResult: repeatable-pose scene, exclude hand/contact
    # uncertainty rather than treating the cut at a gripper as a physical edge.
    gripper_depth = s['empty_depth'].copy()
    gripper_depth[hand_union] = 1500
    current_depth = depth.copy()
    current_depth[hand_union] = 1500
    static = np.zeros_like(obj)
    static[:10,80:160] = True
    segmenter = ReferenceGripperSegmenter()
    reference = segmenter.calibrate(s['empty_depth'],gripper_depth,
        {'left':(3,180,18,215),'right':(220,180,240,215)},static,
        pose_key='synthetic-outline',registered_to_depth=True)
    segmented = segmenter.segment(current_depth,reference,
        pose_key='synthetic-outline',registered_to_depth=True)
    reference_observed = (segmented.observed_mask & ~segmented.uncertain_mask &
                          ~segmented.robot_mask & ~segmented.hand_masks['left'] &
                          ~segmented.hand_masks['right'])
    reports['ReferenceSegmentationResult'] = match_mask(
        target,segmented.object_mask,observed_mask=reference_observed,**options)

    # 8. BoundaryProximityResult retains the source mask/domain. Its kept paths
    # and near_mask are internal-boundary evidence, not a new exterior.
    proximity = boundary.filter_outer_proximity(observed_mask=observed)
    reports['BoundaryProximityResult'] = match_mask(target,proximity.object_mask,
        observed_mask=proximity.observed_mask,**options)

    # 9-11. Geometry/pick detectors retain object masks. Depth-measurement failure
    # does not by itself invalidate an independently observed color silhouette.
    # Keep the original silhouette domain; pass ~exclude_mask too if cuts were made.
    surface = DepthSurfaceDetector().detect(depth,obj,observed_mask=observed)
    reports['SurfaceResult'] = match_mask(target,surface.object_mask,
        observed_mask=observed,**options)
    bulge = OuterEdgeBulgeDetector().detect(depth,obj,depth_scale_m=.001,
        registered_to_mask=True,observed_mask=observed)
    reports['EdgeBulgeResult'] = match_mask(target,bulge.object_mask,
        observed_mask=observed,**options)
    pick = OverheadPickCandidateDetector().detect(depth,outline,
        background_depth=s['empty_depth'],depth_scale_m=.001,registered_to_mask=True)
    reports['PickCandidateResult'] = match_mask(target,pick.object_mask,
        observed_mask=outline.observed_mask,**options)  # Selected component only.

    # 12. BridgeDrapeResult.foreground_mask is usable ONLY when it represents the
    # intended object (supplied object mask in this example). Background difference
    # can also include hands/clutter. Do not use cropped work/bridge-corridor masks.
    bridge_model = BridgeDrapeDetector()
    bridge = bridge_model.detect_mask(depth,obj,depth_unit='mm',observed_mask=observed)
    reports['BridgeDrapeResult'] = match_mask(target,bridge.foreground_mask,
        observed_mask=observed & ~bridge.invalid_mask,**options)
    color_geometry = analyze_color_geometry(bridge_model,depth,colors,
        depth_unit='mm',observed_mask=color_observed)
    reports['analyze_color_geometry -> BridgeDrapeResult'] = match_mask(
        target,color_geometry.foreground_mask,
        observed_mask=color_observed & ~color_geometry.invalid_mask,**options)

    # 13. LiftResult exposes the input object mask only with diagnostics=True.
    # diagnostics.valid_mask is object-restricted depth support, not observation.
    intrinsics = CameraIntrinsics(250,250,130,120)
    table = TablePlane((0,0,-1),2.)
    lift = LiftDetector().detect(depth*.001,intrinsics,table,obj,
        observed_mask=observed,diagnostics=True)
    reports['LiftResult.diagnostics.object_mask'] = match_mask(
        target,lift.diagnostics['object_mask'],observed_mask=observed,**options)

    # 14. GraspResult can expose an eligible object mask with diagnostics=True.
    # Ambiguous label cuts must stay unobserved. Contact masks are NOT silhouettes.
    grasp = IndependentGraspDetector().detect(current_depth,obj,s['hands'],
        depth_unit='mm',registered_to_depth=True,observed_mask=observed,diagnostics=True)
    grasp_observed = (~grasp.diagnostics['caller_unobserved_mask'] &
                      ~grasp.diagnostics['ambiguous_label_mask'])
    reports['GraspResult.diagnostics.eligible_object_mask'] = match_mask(
        target,grasp.diagnostics['eligible_object_mask'],observed_mask=grasp_observed,**options)

    # ObjectOnlyGraspResult has no reusable whole-object output mask. Keep the
    # original input alongside it; valid_reference/elevated masks are subsets.
    object_grasp = ObjectOnlyGraspDetector().detect(depth,obj,depth_unit='mm',
        observed_mask=observed,diagnostics=True)
    reports['ObjectOnlyGraspDetector retained INPUT mask'] = match_mask(
        target,obj,observed_mask=observed,**options)

    # Ordered closed-contour routes. Never substitute the corner candidate list,
    # boundary lines/topology/measure_components trails, or an open segment.
    # OuterContourResult's normal method above is preferred: it handles competing
    # components and source status together. These return PER-CONTOUR reports.
    direct = {
        'BoundaryResult.exterior': match_exteriors(target,boundary.exterior,
            observed_mask=observed,**options),
        'CornerResult.diagnostics.contours_xy': match_exteriors(
            target,corner.diagnostics['contours_xy'],observed_mask=observed,**options),
        'OuterContourResult.contours.points_xy': match_exteriors(
            target,[c['points_xy'] for c in outline.contours],
            observed_mask=outline.observed_mask,
            source_statuses=[c['status'] for c in outline.contours],**options),
    }
    return dict(scene=s, reports=reports, per_contour=direct,
                object_grasp_state=object_grasp.held)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(exist_ok=False)
    outputs = run_examples()
    records = []
    for name, result in outputs['reports'].items():
        scope = 'single_contour_only' if name == 'closed_xy' else 'joint_search_over_supplied_components'
        records.append(dict(source=name,scope=scope,result=result.to_dict()))
    for name, rows in outputs['per_contour'].items():
        for row in rows:
            records.append(dict(source=name,source_index=row['source_index'],scope='single_contour_only',
                reason=row['reason'],result=None if row['result'] is None else row['result'].to_dict()))
    for i, record in enumerate(records):
        result = record['result']
        print(f"{record['source']}: {result['status'] if result else record['reason']}")
        if args.output_dir is not None and result is not None:
            # Save through the original result, retaining existing PNG safeguards.
            original = (outputs['reports'][record['source']] if 'source_index' not in record else
                        outputs['per_contour'][record['source']][record['source_index']]['result'])
            original.save_debug(args.output_dir/f'{i:02d}.png',outputs['scene']['color'])
    if args.output_dir is not None:
        (args.output_dir/'results.json').write_text(json.dumps(dict(
            evidence='synthetic_API_integration_not_real_accuracy',records=records),
            indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
