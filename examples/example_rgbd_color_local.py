"""Reproducible synthetic comparison, not a real-camera accuracy benchmark.

Run from the checkout with: PYTHONPATH=src python examples/example_rgbd_color_local.py
Replace the arrays below with registered current/empty RGB-D in actual use.
"""
import time
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter


def make_scene(kind, seed=0, shape=(60, 100)):
    h, w = shape
    background_color = np.full((h, w, 3), 140, np.uint8)
    background_depth = np.ones((h, w), float)
    color = background_color.copy()
    obj = np.zeros((h,w), bool); shadow = obj.copy()
    obj[h//5:4*h//5,w//10:4*w//10] = True
    shadow[h//5:4*h//5,4*w//10:7*w//10] = True
    rng = np.random.default_rng(seed)
    color[obj] = 125 if kind == 'weak_color' else 80
    color[shadow] = 80
    if kind == 'mottled':
        color[obj] = np.clip(80+rng.normal(0,8,(obj.sum(),1)),0,255).astype(np.uint8)
    delta = rng.normal(0,.004,(h,w))
    delta[obj] += .020 if kind == 'thick' else .003
    if kind == 'coherent_bias':
        delta[shadow] += .003  # Indistinguishable from a shallow object.
    return color, background_depth-delta, background_color, background_depth, obj, shadow


def run(color, depth, background_color, background_depth, **settings):
    return RGBDBackgroundSegmenter(**settings).detect(
        color, depth, background_color=background_color,
        background_depth=background_depth, color_order='RGB',
        registered_to_depth=True, depth_scale_m=1., diagnostics=True)


if __name__ == '__main__':
    data = make_scene('weak_color')
    result = run(*data[:4], fusion_mode='color_local',
                 local_support_radius_px=4, local_mean_depth_m=.0015,
                 local_min_depth_fraction=.18, local_growth_radius_px=2,
                 local_max_reference_separation_m=.10)
    print('object pixels:', result.object_mask.sum())
    print('inferred pixels:', result.inferred_foreground_mask.sum())
    print('local seeds:', result.metrics['local_seed_pixels'])
    print('case, mode, mean object recall, mean shadow acceptance (10 seeds)')
    for case in ('weak_color','same_color','mottled','thick','coherent_bias'):
        for mode in ('color_region','color_supported','depth_anchored','color_local'):
            scores=[]
            for seed in range(10):
                c,z,bc,bz,obj,shadow=make_scene(case,seed)
                r=run(c,z,bc,bz,fusion_mode=mode)
                scores.append([r.object_mask[obj].mean(),r.object_mask[shadow].mean()])
            a,b=np.mean(scores,axis=0)
            print(f'{case}, {mode}, {a:.3f}, {b:.3f}')
    c,z,bc,bz,*_=make_scene('same_color',shape=(480,640))
    run(c,z,bc,bz,fusion_mode='color_local')
    samples=[]
    for _ in range(3):
        start=time.perf_counter();run(c,z,bc,bz,fusion_mode='color_local')
        samples.append(time.perf_counter()-start)
    print('640x480 median seconds, diagnostics enabled:',np.median(samples))
