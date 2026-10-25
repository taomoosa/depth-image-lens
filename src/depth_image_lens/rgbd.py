"""Feature-wise geometry on color candidates, keeping semantic uncertainty explicit."""
from dataclasses import replace
import numpy as np


def analyze_color_geometry(detector, depth, colors, hand_rois=None, *, depth_unit,
                           arm_mask=None, observed_mask=None, work_roi=None,
                           hand_visibility=("visible","visible"), down=(0,1),
                           depth_jump_m=.10, diagnostics=False, depth_geometry=None):
    """Analyze usable color evidence even when one appearance class is unavailable.

    colors must come from explicitly registered SeededColorAssistant inputs.
    A single-class similarity mask is a hypothesis, not hand/object separation.
    depth_geometry may retain independent foreground geometry when colors fail.
    """
    if colors.summary.get('registration')!='caller_declared':
        raise ValueError('explicitly registered color candidates required')
    candidate=colors.object_appearance_candidate_mask
    if candidate is None:candidate=colors.object_mask
    exclusion=np.zeros(candidate.shape,bool) if arm_mask is None else arm_mask
    if colors.hand_exclusion_mask is not None:exclusion=exclusion|colors.hand_exclusion_mask
    result=detector.detect_mask(depth,candidate,hand_rois,depth_unit=depth_unit,
        observed_mask=observed_mask,arm_mask=exclusion,work_roi=work_roi,
        hand_visibility=hand_visibility,down=down,depth_jump_m=depth_jump_m,diagnostics=diagnostics)
    separated=bool(colors.summary['usable_object_seed_pixels'] and colors.summary['usable_hand_seed_pixels'])
    stages={**result.stage_results,'appearance':{
        'status':'computed' if colors.summary['usable_object_seed_pixels'] else 'insufficient_input',
        'reference':'two_seed_class_candidates' if separated else 'single_seed_class_similarity_hypothesis',
        'class_separation_available':separated,'semantic_identity_verified':False,
        'uncertain_pixels':colors.summary['uncertain_pixels'],
        'warnings':[] if separated else ['object_hand_identity_unresolved']}}
    if depth_geometry is not None:stages['depth_candidate_geometry']=depth_geometry.stage_results
    return replace(result,stage_results=stages,features={**result.features,
        'rgb_geometry_scope':'appearance_candidate_hypothesis_not_semantic_or_grasp_truth'})
