"""Repeatable-pose gripper references, not free-view semantic segmentation."""
from dataclasses import dataclass, field
import numpy as np
from .debug import DebugResultMixin
from .depth import _numeric_image, _validate_depth_scale
from .geometry import bool_mask


def _color(image, shape, order):
    if order not in ('RGB','BGR'):raise ValueError('explicit RGB or BGR color_order required')
    a=np.asarray(image)
    if a.dtype!=np.uint8 or a.shape!=(*shape,3):raise ValueError('aligned uint8 HxWx3 color required')
    return np.array(a if order=='RGB' else a[...,::-1],copy=True)


def _dilate(mask):
    h,w=mask.shape;p=np.pad(mask,1);out=np.zeros_like(mask)
    for y in range(3):
        for x in range(3):out|=p[y:y+h,x:x+w]
    return out


@dataclass(frozen=True)
class GripperReference(DebugResultMixin):
    empty_depth_m: np.ndarray = field(repr=False)
    gripper_depth_m: np.ndarray = field(repr=False)
    hand_masks: dict = field(repr=False)
    static_mask: np.ndarray = field(repr=False)
    search_masks: dict = field(repr=False)
    robot_mask: np.ndarray = field(repr=False)
    reference_valid: np.ndarray = field(repr=False)
    pose_key: str
    parameters: dict
    color_rgb: np.ndarray | None = field(default=None,repr=False)
    def to_dict(self):
        return dict(pose_key=self.pose_key,parameters=self.parameters,
            hand_pixels={k:int(v.sum()) for k,v in self.hand_masks.items()},
            reference_invalid_pixels=int((~self.reference_valid).sum()),static_pixels=int(self.static_mask.sum()),robot_pixels=int(self.robot_mask.sum()),has_color=self.color_rgb is not None,
            scope='caller_named_repeatable_pose_reference_not_semantic_truth')


@dataclass(frozen=True)
class ReferenceSegmentationResult(DebugResultMixin):
    object_mask: np.ndarray = field(repr=False)
    hand_masks: dict = field(repr=False)
    uncertain_mask: np.ndarray = field(repr=False)
    foreground_mask: np.ndarray = field(repr=False)
    robot_mask: np.ndarray = field(repr=False)
    observed_mask: np.ndarray = field(repr=False)
    summary: dict
    diagnostics: dict = field(default_factory=dict,repr=False)
    def to_dict(self):return dict(self.summary)


class ReferenceGripperSegmenter:
    """Calibrate once per fixed camera/pose; isolate runtime depth/color changes.

    Coarse end-effector boxes and a static background ROI are required at setup,
    not perfect per-frame masks. Pose or scene mismatch causes abstention.
    A surface imitating the reference in depth and color is indistinguishable.
    """
    def __init__(self, *, foreground_m=.02, match_depth_m=.01, match_color_distance=45.,
                 static_depth_m=.01, static_color_distance=35., min_pixels=20,
                 min_valid_fraction=.7, min_match_fraction=.7, max_unexpected_fraction=.05):
        values=[foreground_m,match_depth_m,match_color_distance,static_depth_m,static_color_distance,
                min_valid_fraction,min_match_fraction,max_unexpected_fraction]
        if not np.isfinite(values).all() or min(values)<=0 or min_valid_fraction>1 or min_match_fraction>1 or max_unexpected_fraction>1:
            raise ValueError('positive finite thresholds and fractions <=1 required')
        if type(min_pixels) is not int or min_pixels<3:raise ValueError('min_pixels integer >=3 required')
        self.parameters=dict(foreground_m=float(foreground_m),match_depth_m=float(match_depth_m),
            match_color_distance=float(match_color_distance),static_depth_m=float(static_depth_m),
            static_color_distance=float(static_color_distance),min_pixels=min_pixels,
            min_valid_fraction=float(min_valid_fraction),min_match_fraction=float(min_match_fraction),
            max_unexpected_fraction=float(max_unexpected_fraction))

    def calibrate(self, empty_depth, gripper_depth, hand_rois, static_mask, *, pose_key,
                  depth_scale_m=.001, gripper_color=None, color_order='RGB', registered_to_depth):
        """Empty scene and end-effectors-only reference; same camera/pixels/units.

        hand_rois maps left/right to half-open (x0,y0,x1,y1) coarse end-effector
        boxes. Arms/background inside them can contaminate the reference: inspect
        its debug view before use. No object may be present during capture.
        """
        if registered_to_depth is not True:raise ValueError('registered_to_depth=True required')
        _validate_depth_scale(depth_scale_m)
        if not isinstance(pose_key,str) or not pose_key:raise ValueError('nonempty pose_key required')
        empty=_numeric_image(empty_depth,'empty_depth')*depth_scale_m
        gripper=_numeric_image(gripper_depth,'gripper_depth')*depth_scale_m
        if empty.shape!=gripper.shape:raise ValueError('reference depth shapes must match')
        if not isinstance(hand_rois,dict) or set(hand_rois)!={'left','right'}:raise ValueError('left/right hand_rois required')
        valid=np.isfinite(empty)&np.isfinite(gripper)&(empty>0)&(gripper>0)
        static=bool_mask(static_mask,empty.shape,'static_mask').copy()
        p=self.parameters
        if (static&valid).sum()<p['min_pixels'] or (static&valid).sum()/max(static.sum(),1)<p['min_valid_fraction']:
            raise ValueError('insufficient valid static reference ROI')
        if np.quantile(np.abs(empty[static&valid]-gripper[static&valid]),.9)>p['static_depth_m']:
            raise ValueError('static reference ROI changed during capture')
        with np.errstate(invalid='ignore'):fg=valid&(empty-gripper>p['foreground_m'])
        masks={};search_masks={};boxes=np.zeros(empty.shape,bool)
        h,w=empty.shape
        for side,box in hand_rois.items():
            if len(box)!=4 or any(type(x) is not int for x in box):raise ValueError('ROI must contain four integers')
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('reference ROI must be inside image')
            roi=np.zeros(empty.shape,bool);roi[y0:y1,x0:x1]=True
            if (boxes&roi).any() or (static&roi).any():raise ValueError('hand ROIs and static ROI must not overlap')
            boxes|=roi;search_masks[side]=roi;masks[side]=fg&roi
            if masks[side].sum()<p['min_pixels']:raise ValueError('insufficient gripper foreground in '+side+' ROI')
        rgb=None if gripper_color is None else _color(gripper_color,empty.shape,color_order)
        return GripperReference(empty.copy(),gripper.copy(),masks,static,search_masks,fg&~boxes,valid.copy(),pose_key,dict(p),rgb)

    def segment(self, depth, reference, *, pose_key, depth_scale_m=.001, color=None,
                color_order='RGB', registered_to_depth, observed_mask=None,
                hand_visibility=('visible','visible'), diagnostics=False):
        if registered_to_depth is not True:raise ValueError('registered_to_depth=True required')
        if not isinstance(reference,GripperReference):raise ValueError('GripperReference required')
        if type(diagnostics) is not bool:raise ValueError('diagnostics must be bool')
        if len(hand_visibility)!=2 or any(v not in ('visible','partial','hidden') for v in hand_visibility):raise ValueError('two visibility declarations required')
        _validate_depth_scale(depth_scale_m);z=_numeric_image(depth,'depth')*depth_scale_m
        if z.shape!=reference.empty_depth_m.shape:raise ValueError('depth/reference shape mismatch; recapture after crop/resize')
        obs=np.ones(z.shape,bool) if observed_mask is None else bool_mask(observed_mask,z.shape,'observed_mask')
        valid=obs&np.isfinite(z)&(z>0);bg=np.isfinite(reference.empty_depth_m)&(reference.empty_depth_m>0)
        p=self.parameters
        with np.errstate(invalid='ignore'):
            fg=valid&bg&(reference.empty_depth_m-z>p['foreground_m'])
            dz=np.abs(z-reference.gripper_depth_m)
        rgb=None if color is None else _color(color,z.shape,color_order)
        color_delta=None if rgb is None or reference.color_rgb is None else np.linalg.norm(rgb.astype(float)-reference.color_rgb.astype(float),axis=2)
        failures=[];static=reference.static_mask;sv=static&valid&np.isfinite(reference.gripper_depth_m)
        static_coverage=float(sv.sum()/max(static.sum(),1))
        static_error=float(np.quantile(dz[sv],.9)) if sv.any() else None
        static_color=float(np.quantile(color_delta[sv],.9)) if color_delta is not None and sv.any() else None
        if pose_key!=reference.pose_key:failures.append('pose_key_mismatch')
        if sv.sum()<p['min_pixels'] or static_coverage<p['min_valid_fraction']:failures.append('static_region_unobserved')
        elif static_error>p['static_depth_m']:failures.append('static_scene_depth_changed')
        if reference.color_rgb is not None and rgb is None:failures.append('reference_color_required')
        if static_color is not None and static_color>p['static_color_distance']:failures.append('static_illumination_or_alignment_changed')
        all_expected=reference.hand_masks['left']|reference.hand_masks['right']
        hands={};sides={};uncertain=~valid|~bg|~reference.reference_valid;blocked=np.zeros(z.shape,bool)
        for i,side in enumerate(('left','right')):
            expected=reference.hand_masks[side];usable=expected&valid
            matched=usable&fg&(dz<=p['match_depth_m'])
            if color_delta is not None:matched &= color_delta<=p['match_color_distance']
            coverage=float(usable.sum()/max(expected.sum(),1));fraction=float(matched.sum()/max(usable.sum(),1))
            reference_level=float(np.median(reference.gripper_depth_m[expected]))
            unexpected=reference.search_masks[side]&~expected&fg&(np.abs(z-reference_level)<=p['match_depth_m'])
            if rgb is not None and reference.color_rgb is not None:
                tone=np.median(reference.color_rgb[expected],axis=0)
                unexpected &= np.linalg.norm(rgb.astype(float)-tone,axis=2)<=p['match_color_distance']
            unexpected_fraction=float(unexpected.sum()/max(expected.sum(),1))
            reasons=list(failures)
            if hand_visibility[i]=='hidden':reasons.append('caller_hidden_side')
            if unexpected_fraction>p['max_unexpected_fraction']:reasons.append('unexpected_gripper_like_extent')
            if coverage<p['min_valid_fraction']:reasons.append('reference_region_coverage_low')
            if matched.sum()<p['min_pixels'] or fraction<p['min_match_fraction']:reasons.append('insufficient_pose_reference_match')
            # A poor match may be pose change, occlusion, or segmentation failure.
            # Do not reclassify that whole region as a known object.
            if reasons:blocked|=reference.search_masks[side];matched[:]=False
            hands[side]=matched
            sides[side]=dict(status='candidate' if not reasons else 'unknown',reasons=reasons,
                reference_pixels=int(expected.sum()),observed_pixels=int(usable.sum()),
                reference_search_valid_fraction=float(reference.reference_valid[reference.search_masks[side]].mean()),
                reference_search_unknown_pixels=int((reference.search_masks[side]&~reference.reference_valid).sum()),
                matching_pixels=int(matched.sum()),valid_fraction=coverage,match_fraction=fraction,
                unexpected_gripper_like_pixels=int(unexpected.sum()),unexpected_fraction=unexpected_fraction,
                hand_visibility=hand_visibility[i],physical_identity='caller_side_assignment')
        matched_hands=hands['left']|hands['right']
        # Foreground already present outside hand boxes in the object-free
        # reference is robot/clutter, never newly appearing object material.
        robot_expected=reference.robot_mask
        robot_match=robot_expected&valid&fg&(dz<=p['match_depth_m'])
        if color_delta is not None:robot_match &= color_delta<=p['match_color_distance']
        robot_fraction=float(robot_match.sum()/robot_expected.sum()) if robot_expected.any() else None
        object_failures=[]
        if robot_fraction is not None and robot_fraction<p['min_match_fraction']:
            object_failures.append('robot_reference_match_low_object_identity_unresolved')
        uncertain |= robot_expected&~robot_match
        # New foreground outside the expected hand, or clearly in front of it,
        # is an object candidate only under the declared controlled-scene workflow.
        with np.errstate(invalid='ignore'):
            closer=valid&(reference.gripper_depth_m-z>p['match_depth_m'])
        objects=fg&(~all_expected|closer)&~matched_hands&~blocked&~robot_expected&reference.reference_valid
        uncertain |= fg&~(objects|matched_hands|robot_match)
        contact_band=_dilate(matched_hands)&_dilate(objects)
        uncertain |= contact_band|blocked
        if failures or object_failures:objects[:]=False;uncertain|=fg&~matched_hands
        summary=dict(status='candidate' if not failures and not object_failures else 'unknown',reasons=failures+object_failures,sides=sides,
            reference_invalid_pixels=int((~reference.reference_valid).sum()),robot_reference_pixels=int(robot_expected.sum()),robot_matching_pixels=int(robot_match.sum()),robot_match_fraction=robot_fraction,
            foreground_pixels=int(fg.sum()),object_candidate_pixels=int(objects.sum()),
            uncertain_pixels=int(uncertain.sum()),static_valid_fraction=static_coverage,
            static_depth_p90_m=static_error,static_color_p90=static_color,parameters=dict(p),
            warnings=['repeatable_camera_and_pose_required','depth_color_match_is_not_semantic_identity',
                'identical_depth_color_occlusion_is_unobservable','object_candidates_assume_no_other_moving_clutter',
                'contact_boundary_is_uncertain','mask_completeness_not_established','unreported_robot_motion_may_escape_reference_checks'],
            validation_status='not_validated',physical_grasp='unknown')
        debug=dict(reference_left_mask=reference.hand_masks['left'].copy(),reference_right_mask=reference.hand_masks['right'].copy(),
            static_mask=static.copy(),contact_uncertain_mask=contact_band,blocked_region_mask=blocked) if diagnostics else {}
        return ReferenceSegmentationResult(objects,hands,uncertain,fg,robot_match,valid&bg,summary,debug)
