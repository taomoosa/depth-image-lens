"""Conservative uniform-color shadow hypotheses from declared color samples."""
import numpy as np
from .geometry import bool_mask


def uniform_shadow_selection(rgb,reference,available,check,seed,foreground,uncertain,p):
    """Keep the original foreground unless a distinguishable object model is given."""
    c=rgb.astype(float);b=reference.astype(float);shape=available.shape
    ci=c.mean(axis=2);bi=b.mean(axis=2)
    cc=c/np.maximum(c.sum(axis=2,keepdims=True),1.)
    bc=b/np.maximum(b.sum(axis=2,keepdims=True),1.)
    threshold=p['shadow_chroma_threshold'];floor=p['shadow_intensity_floor']
    usable_c=(ci>=floor)&(c.min(axis=2)>1)&(c.max(axis=2)<254)
    usable_b=(bi>=floor)&(b.min(axis=2)>1)&(b.max(axis=2)<254)
    usable=available&usable_c&usable_b
    chroma=np.linalg.norm(cc-bc,axis=2);ratio=ci/np.maximum(bi,1.)
    candidates=foreground&usable&(chroma<=threshold)&(ratio>=p['shadow_min_ratio'])&(ratio<=p['shadow_max_ratio'])
    suppressed=np.zeros(shape,bool);seed_mask=np.zeros(shape,bool)
    info=dict(status='insufficient',mode='uniform_color_shadow_hypothesis',reasons=[],
              chroma_threshold=threshold,intensity_floor=floor,ratio_limits=[p['shadow_min_ratio'],p['shadow_max_ratio']],
              candidate_pixels=int(candidates.sum()),suppressed_pixels=0,
              warnings=['uniform_single_object_and_empty_registered_reference_are_caller_assumptions',
                        'normalized_encoded_rgb_is_not_calibrated_illumination_invariance',
                        'shadow_or_brightness_only_object_is_ambiguous_without_distinct_object_chromaticity',
                        'suppressed_pixels_remain_uncertain_hypotheses'])
    obj=foreground.copy();unknown=uncertain.copy()|(~usable&available)
    def finish():
        if info['status']!='computed':unknown[foreground]=True
        x=chroma.copy();v=ratio.copy();x[~usable]=np.nan;v[~usable]=np.nan
        debug=dict(shadow_candidate_mask=candidates,shadow_suppressed_mask=suppressed,
                   shadow_unreliable_mask=available&~usable,object_color_seed_mask=seed_mask,
                   shadow_chroma_distance=x,shadow_intensity_ratio=v)
        return obj,unknown,info,debug
    if check is None or (check&usable).sum()<25:
        info['reasons'].append('stable_table_check_requires_25_usable_pixels');return finish()
    if seed is None:
        info['reasons'].append('object_color_seed_required_before_shadow_suppression');return finish()
    try:seed_mask=bool_mask(seed,shape,'object_color_seed').copy()
    except (ValueError,TypeError):
        info['reasons'].append('invalid_object_color_seed');return finish()
    sample=seed_mask&usable_c
    info['seed_pixels']=int(seed_mask.sum());info['usable_seed_pixels']=int(sample.sum())
    if sample.sum()<25 or sample.sum()<.9*seed_mask.sum():
        info['reasons'].append('object_seed_requires_25_usable_pixels_and_90_percent_coverage');return finish()
    if (seed_mask&check).any():
        info['reasons'].append('object_seed_overlaps_declared_table_check');return finish()
    good_ref=available&usable_b
    if good_ref.sum()<25 or good_ref.sum()<.8*available.sum():
        info['reasons'].append('insufficient_unclipped_reference_support');return finish()
    table=np.median(bc[good_ref],axis=0);model=np.median(cc[sample],axis=0)
    spread_ref=float(np.quantile(np.linalg.norm(bc[good_ref]-table,axis=1),.95))
    spread_obj=float(np.quantile(np.linalg.norm(cc[sample]-model,axis=1),.95))
    separation=float(np.linalg.norm(model-table))
    info.update(reference_chroma_median=table.tolist(),object_chroma_median=model.tolist(),
                reference_spread_p95=spread_ref,object_seed_spread_p95=spread_obj,model_separation=separation)
    if spread_ref>threshold:info['reasons'].append('reference_not_approximately_uniform_chromaticity')
    if spread_obj>threshold:info['reasons'].append('object_seed_not_approximately_uniform_chromaticity')
    if separation<=2*threshold+spread_ref+spread_obj:info['reasons'].append('object_and_table_chromaticity_not_separable')
    if info['reasons']:return finish()
    object_distance=np.linalg.norm(cc-model,axis=2)
    suppressed[:]=candidates&(object_distance>chroma+threshold)
    obj[suppressed]=False
    unknown|=suppressed|(foreground&((object_distance>threshold)|~usable))
    info.update(status='computed',suppressed_pixels=int(suppressed.sum()),
                retained_foreground_pixels=int(obj.sum()),
                reasons=['shadow_suppression_requires_declared_single_uniform_object_color'])
    return finish()
