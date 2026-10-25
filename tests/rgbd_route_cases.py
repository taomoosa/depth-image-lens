"""Two new fixed geometries; no reuse of original or missingness scene builders."""
import numpy as np


def route_cases():
    for family in ('curved', 'sloped'):
        for condition in ('clean', 'unknown_branch', 'partial_unknown_patch', 'full_semantic_cut',
                          'unknown_robot_bridge', 'known_robot_bridge', 'raw_depth_cut', 'cropped_anchor'):
            h,w=132,178; obj=np.zeros((h,w),bool); hand=np.zeros_like(obj)
            xs=np.arange(30,147); t=(xs-30)/116
            tops=(31+np.rint(22*np.sin(np.pi*t)**2 if family=='curved' else 8*t)).astype(int)
            for x,y in zip(xs,tops):obj[y:y+14,x]=True
            right_y=int(tops[-1]); boxes={'left':(23,12,40,33),'right':(138,right_y-19,155,right_y+2)}
            hand[17:28,25:37]=True;hand[right_y-14:right_y-3,140:152]=True
            patch=np.zeros_like(obj);truth='yes';drape='both' if family=='curved' else 'neither'
            if condition=='unknown_branch':
                y=int(tops[58])+14; obj[y-1:y+17,85:92]=True;patch[y:y+17,85:92]=True
            if condition=='partial_unknown_patch':
                y=int(tops[58]);patch[y:y+4,86:91]=True
            if condition=='full_semantic_cut':patch[:,86:91]=True
            if condition in ('unknown_robot_bridge','known_robot_bridge'):
                bar=obj.copy();obj[:]=False;obj[32:92,28:37]=True
                hand|=bar&~obj;truth='no';drape='left'
                if condition=='unknown_robot_bridge':patch|=bar&~obj
            fg=obj|hand;depth=np.where(fg,1430,2300).astype(np.uint16);diff=np.where(fg,870,0).astype(np.uint16)
            rgb=np.full((h,w,3),35,np.uint8);rgb[obj]=(35,115,205);rgb[hand]=(195,125,65)
            rgb[patch&fg]=(220,35,150)
            os=np.zeros_like(obj);hs=np.zeros_like(obj);os[34,32]=True;hs[19,28]=True
            if condition=='raw_depth_cut':depth[:,86:91]=0
            if condition=='cropped_anchor':boxes['right']=(171,right_y-19,188,right_y+2)
            yield dict(name=family+'_'+condition,group='independent_route_validation',condition=condition,family=family,
                       depth=depth,difference=diff,color=rgb,object_seeds=os,hand_seeds=hs,hand_rois=boxes,
                       object_gt=obj,bridge_truth=truth,drape_truth=drape,
                       provenance='fixed two-family semantic-route construction; correlated conditions; not real GT')
