"""Fixed small synthetic comparison, not real-image accuracy or a tuning sweep."""
import json
from pathlib import Path
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from .test_bridge_drape import scene, inputs, ROIS


def cases():
    out=[]
    def add(name,m,truth,**kw):out.append((name,m,truth,kw))
    add('flat',scene(),'yes');add('deep_u',scene('u'),'yes')
    for side in ('left','right'):
        m=scene(side);x=100 if side=='left' else 20;m[13:21,x-4:x+5]=True
        add(side+'_visible_opposite',m,'no')
        add(side+'_opposite_absent',scene(side),'no',opposite_absent=True)
    add('separate_drapes',scene('separate'),'no')
    arm=np.zeros((100,120),bool);arm[7:12,20:101]=True;arm[7:20,17:24]=True;arm[7:20,97:104]=True
    add('arms_connected_above_band',arm,'no')
    m=scene();m[20:80,17:24]=True;add('bridge_asymmetric_drape',m,'yes')
    for delta,name in [(8,'sloped_height_within_band'),(30,'height_assumption_violation')]:
        m=np.zeros((100,120),bool)
        for x in range(20,101):
            y=20+round(delta*(x-20)/80);m[y:y+7,x]=True
        add(name,m,'yes',delta=delta)
    add('missing_depth_cut',scene(),'yes',missing=True)
    m=scene();m[:,55:65]=False;add('large_observed_gap',m,'no')
    m=np.zeros((100,120),bool);m[:50,17:24]=True;add('upper_cut',m,'no')
    add('empty',np.zeros((100,120),bool),'no')
    # Identical pixels can be a deformable-object bridge or an arm slab. Keep this error in totals.
    add('arm_slab_without_semantic_exclusion',scene(),'no',ambiguous_arm=True)
    add('arm_slab_with_exclusion',scene(),'no',exclude=True)
    add('depth_step_bridge',scene(),'yes',depth_step=True)
    return out


def run():
    detector=BridgeDrapeDetector();rows=[]
    for name,mask,truth,kw in cases():
        z,diff=inputs(mask);boxes=dict(ROIS);opts={}
        if 'delta' in kw:
            delta=kw['delta'];boxes['right']=(94,4+delta,106,20+delta)
        if kw.get('missing'):z[:,57:64]=0
        if kw.get('exclude'):opts['arm_mask']=mask
        if kw.get('depth_step'):
            z[mask&(np.indices(mask.shape)[1]>60)]=1.1;diff=2.-z
        manual=detector.detect(z,diff,boxes,**opts)
        auto=detector.detect_auto(z,diff,upper_band=(8,20),**opts)
        # Opposite absence is not an empty, semantically identified hand ROI in auto.
        errors=None
        if name in ('flat','deep_u','left_visible_opposite','right_visible_opposite','separate_drapes','bridge_asymmetric_drape','sloped_height_within_band') and 'anchor_points_uv' in auto.auto:
            gt=np.array([[19.5,19.],[99.5,19.+kw.get('delta',0)]])
            estimated=np.array(auto.auto['anchor_points_uv']);errors=np.linalg.norm(estimated-gt,axis=1).tolist()
        rows.append({'scene':name,'synthetic_bridge_GT':truth,'manual':manual.to_dict(),'auto':auto.to_dict(),
                     'anchor_point_error_px_synthetic_only':errors,'semantic_ambiguity':bool(kw.get('ambiguous_arm'))})
    summary={}
    for mode in ('manual','auto'):
        counts={'match':0,'false_bridge':0,'missed_bridge':0,'unknown':0}
        for r in rows:
            pred=r[mode]['bridge'];gt=r['synthetic_bridge_GT']
            if pred=='unknown':counts['unknown']+=1
            elif pred==gt:counts['match']+=1
            elif pred=='yes':counts['false_bridge']+=1
            else:counts['missed_bridge']+=1
        summary[mode]=counts
    result={'scenes':18,'synthetic_only':True,'no_tuning_sweep':True,'manual_ROIs_include_scene_support_locations':True,
        'auto_inputs':'same depth/difference; upper_band8..20; no hand ROIs; arm mask only for explicit exclusion case',
        'parameters':{'min_span_px':30,'max_height_delta_px':12,'depth_jump_m':.10},'bridge_counts':summary,'rows':rows}
    return result


if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
