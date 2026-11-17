"""Runnable synthetic millimeter-depth example; replace arrays with registered captures."""
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter

background_color = np.full((400, 640, 3), 150, np.uint8)
background_depth = np.full((400, 640), 1000, np.uint16)
color = background_color.copy()
depth = background_depth.copy()
color[100:300, 160:480] = (70, 100, 160)
depth[100:300, 160:480] = 998  # 2 mm; no pixels exceed default 5 mm.
segmenter = RGBDBackgroundSegmenter(
    fusion_mode="color_first",
    first_min_depth_delta_m=0.00075,
    first_noise_multiplier=2.5,
    local_max_reference_separation_m=0.05,
)
result = segmenter.detect(
    color, depth, background_color=background_color,
    background_depth=background_depth, color_order="RGB",
    registered_to_depth=True, depth_scale_m=0.001, diagnostics=True,
)
print("Selected pixels:", int(result.object_mask.sum()))
print("Region evidence:", result.stages["fusion"]["regions"])
# result.save_debug("color-first.png", color=color, color_order="RGB")
# Inferred pixels are not verified material or grasp-safe measured support.
