"""Small geometry fixtures for focused automatic-anchor regressions."""
import numpy as np
from .test_bridge_drape import scene


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
    # Identical pixels can be a deformable-object bridge or an arm slab. Keep this indistinguishable-input regression.
    add('arm_slab_without_semantic_exclusion',scene(),'no',ambiguous_arm=True)
    add('arm_slab_with_exclusion',scene(),'no',exclude=True)
    add('depth_step_bridge',scene(),'yes',depth_step=True)
    return out
