"""Frozen new geometries for local-support evaluation, not independent captures."""
import numpy as np


def cases():
    import cv2
    rng=np.random.default_rng(24519)
    specifications=[(f'rect_{w}_{h}_r{a}',w,h,a) for w,h in ((19,27),(24,37),(31,46),(43,55)) for a in (7,23,39)]
    specifications += [(name,60,70,0) for name in ('rounded_small','rounded_large','concave','crop','components','holes','step')]
    for name,w,h,degrees in specifications:
        mask=np.zeros((160,200),np.uint8)
        center=np.array([80,80])+rng.integers(-8,9,size=2)
        angle=np.deg2rad(degrees);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        polygon=np.rint((np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])-[(w-1)/2,(h-1)/2])@R.T+center).astype(int)
        truth=polygon.copy();hazard=np.empty((0,2));hidden=np.empty((0,2))
        if name.startswith('rounded'):
            size=(12,17) if name.endswith('small') else (30,35)
            cv2.ellipse(mask,tuple(center),size,0,0,360,1,-1);truth=np.empty((0,2))
        else:
            if name=='concave':
                polygon=np.array([[40,40],[110,40],[110,70],[75,70],[75,120],[40,120]])
                truth=polygon[[0,1,2,4,5]]
            if name=='crop':
                polygon=np.array([[-15,40],[45,40],[45,110],[-15,110]]);truth=polygon[[1,2]]
            if name=='components':
                polygon=np.array([[50,50],[104,50],[104,109],[50,109]])
                second=np.array([[18,18],[45,18],[45,46],[18,46]])
                cv2.fillPoly(mask,[second],1);truth=np.concatenate([polygon,second])
            cv2.fillPoly(mask,[polygon],1)
        mask=mask.astype(bool);depth=np.where(mask,980,1000).astype(np.uint16)
        if name=='components':depth[18:47,18:46]=920
        if name=='holes':
            x,y=polygon[0];depth[max(0,y-5):y+25,max(0,x-5):x+25]=0
        if name=='step':
            x,y=polygon[0];depth[mask&(np.indices(mask.shape)[1]<x+12)]=920;hazard=polygon[[0,3]]
        color=np.full((*mask.shape,3),60,np.uint8);color[mask]=[40,100,190]
        yield {'name':name,'mask':mask,'depth':depth,'color':color,'truth':truth,'observable_step_corners':hazard}
