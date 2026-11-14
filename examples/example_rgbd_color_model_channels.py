"""Synthetic brightness/chroma tuning sweep; not real-camera accuracy."""
import numpy as np
import cv2
from depth_image_lens import RGBDBackgroundSegmenter


def scene(core=(110,170,145), edge=(145,170,145), shadow=(80,128,128),
          background=(160,128,128), height_mm=8, depth_noise_mm=0, seed=0):
    lab=np.empty((64,90,3),np.uint8);lab[:]=background;reference=lab.copy()
    lab[12:52,12:40]=core;lab[12:52,40:48]=edge;lab[12:52,48:56]=shadow
    color=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    background_color=cv2.cvtColor(reference,cv2.COLOR_LAB2RGB)
    depth=np.full((64,90),1.,float);background_depth=depth.copy()
    depth[12:52,12:40]-=height_mm/1000
    if depth_noise_mm:
        depth+=np.random.default_rng(seed).normal(0,depth_noise_mm/1000,depth.shape)
    return color,depth,background_color,background_depth


def evaluate(data, luma_tolerance=None, chroma_tolerance=None, **overrides):
    color,depth,background_color,background_depth=data
    parameters=dict(
        fusion_mode='color_model',depth_threshold_m=.002,depth_margin_m=.0005,
        model_color_tolerance=18,model_luma_tolerance=luma_tolerance,
        model_chroma_tolerance=chroma_tolerance,model_background_margin=1,
        model_boundary_radius_px=10)
    parameters.update(overrides)
    segmenter=RGBDBackgroundSegmenter(**parameters)
    return segmenter.detect(color,depth,background_color=background_color,
        background_depth=background_depth,color_order='RGB',depth_scale_m=1.,
        registered_to_depth=True,diagnostics=True)


def fractions(result):
    return [float(result.object_mask[12:52,a:b].mean()) for a,b in
            ((12,40),(40,48),(48,56))]


def main():
    cases={
        'colored_bright_edge':{},
        'gray_bright_edge':dict(core=(100,128,128),edge=(125,128,128)),
        'gray_edge_near_background':dict(core=(100,128,128),edge=(148,128,128)),
        'same_color_shadow':dict(shadow=(145,170,145)),
        'different_chroma_shadow':dict(shadow=(100,150,138)),
        'thin_noisy':dict(height_mm=3,depth_noise_mm=2),
        'no_depth_support':dict(height_mm=0),
    }
    print('case; luma tolerance; core recall / edge recall / shadow acceptance')
    for name,kwargs in cases.items():
        for tolerance in (None,36,60):
            values=[fractions(evaluate(scene(**kwargs,seed=seed),tolerance,18))
                    for seed in range(10)]
            print(name,tolerance,np.round(np.mean(values,axis=0),3).tolist(),sep='; ')
    print('negative depth edge; policy; core / edge / shadow')
    data=list(scene(edge=(110,170,145)));data[1][12:52,40:48]=1.003
    for policy in ({},{'model_farther_threshold_m':.004},{'model_reject_farther':False}):
        print('negative_depth',policy,np.round(fractions(evaluate(data,**policy)),3).tolist(),sep='; ')
    print('background-closer gray edge; margin; core / edge / shadow')
    data=scene(core=(100,128,128),edge=(135,128,128))
    for margin in (1,0,-1,-4):
        print('negative_margin',margin,np.round(fractions(evaluate(data,60,18,
              model_background_margin=margin)),3).tolist(),sep='; ')


if __name__=='__main__':main()
