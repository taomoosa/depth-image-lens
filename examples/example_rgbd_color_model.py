"""Fixed region colors and background competition: synthetic example/evaluation.

Run: PYTHONPATH=src python examples/example_rgbd_color_model.py
This is not a real-camera benchmark or an automatic parameter calibration.
"""
import time
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter


def scene(kind, seed=0, shape=(90,140)):
    h,w=shape;bc=np.full((h,w,3),140,np.uint8);bz=np.ones((h,w))
    c=bc.copy();obj=np.zeros((h,w),bool);shadow=obj.copy();edge=obj.copy()
    y0,y1=h//6,5*h//6;x0,x1=w//10,w//2
    obj[y0:y1,x0:x1]=True;shadow[y0:y1,x1:4*w//5]=True
    edge[obj]=True;edge[y0+5:y1-5,x0+5:x1-5]=False
    c[obj]=80;c[edge]=95;c[shadow]=80 if kind=='same_color_shadow' else 115
    rng=np.random.default_rng(seed)
    d=rng.normal(0,.001,(h,w));d[obj]+=.006;d[edge]-=.0056
    if kind=='thin_noisy':
        d=rng.normal(0,.004,(h,w));d[obj]+=.003
    if kind=='mottled':
        c[obj]=np.clip(80+rng.normal(0,5,(obj.sum(),1)),0,255).astype(np.uint8)
    if kind=='coherent_false_seed':
        d[shadow]+=.006
    return c,bz-d,bc,bz,obj,shadow,edge


def detect(mode, data, radius=6):
    c,z,bc,bz=data[:4]
    return RGBDBackgroundSegmenter(
        fusion_mode=mode,depth_threshold_m=.002,depth_margin_m=.0005,
        local_seed_min_area=256,local_background_depth_m=.00025,
        model_boundary_radius_px=radius,
    ).detect(c,z,background_color=bc,background_depth=bz,color_order='RGB',
             registered_to_depth=True,depth_scale_m=1.,diagnostics=True)


if __name__=='__main__':
    r=detect('color_model',scene('edge_dropout'))
    print('selected:',int(r.object_mask.sum()))
    print('inferred:',int(r.inferred_foreground_mask.sum()))
    print('models:',r.stages['fusion']['models'])
    print('case, mode, object recall, edge recall, shadow acceptance (10 fixed seeds)')
    for case in ('edge_dropout','same_color_shadow','mottled','thin_noisy','coherent_false_seed'):
        for mode in ('color_local','color_model'):
            values=[]
            for seed in range(10):
                data=scene(case,seed);result=detect(mode,data)
                values.append([result.object_mask[m].mean() for m in data[4:]])
            obj,shadow,edge=np.mean(values,axis=0)
            print(f'{case}, {mode}, {obj:.3f}, {edge:.3f}, {shadow:.3f}')
    print('radius sweep: radius, edge recall, shadow acceptance, total grid perimeter')
    for radius in (2,6,10):
        values=[]
        for seed in range(10):
            data=scene('same_color_shadow',seed);r=detect('color_model',data,radius)
            m=r.object_mask.astype(int)
            perimeter=np.abs(np.diff(m,axis=0)).sum()+np.abs(np.diff(m,axis=1)).sum()
            values.append([r.object_mask[data[6]].mean(),r.object_mask[data[5]].mean(),perimeter])
        print(radius, np.round(np.mean(values,axis=0),3).tolist())
    data=scene('edge_dropout',shape=(480,640))
    for mode in ('color_local','color_model'):
        detect(mode,data);samples=[]
        for _ in range(3):
            t=time.perf_counter();detect(mode,data);samples.append(time.perf_counter()-t)
        print(mode,'640x480 median seconds:',float(np.median(samples)))
