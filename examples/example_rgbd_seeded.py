"""Minimal bounded RGB-D color growth example; no field-accuracy claim."""
import json
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector


# Synthetic observations: a strong-color center with a weak-color rim.
empty_color = np.full((32, 40, 3), 140, np.uint8)
empty_depth = np.full((32, 40), 1000, np.uint16)
color, depth = empty_color.copy(), empty_depth.copy()
color[7:25, 8:32] = (110, 145, 165)
color[9:23, 10:30] = (30, 150, 210)
depth[7:25, 8:32] = 980
# A genuine background hole stays a hole: neither depth nor color supports growth.
color[14:18, 18:22] = empty_color[14:18, 18:22]
depth[14:18, 18:22] = 1000

seg = RGBDBackgroundSegmenter(
    fusion_mode='color_seeded',
    color_relaxation_ratio=.5,
    color_growth_max_steps=2,
).detect(color, depth, background_color=empty_color, background_depth=empty_depth,
         color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
         diagnostics=True)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.outline_observed_mask)
print(json.dumps(dict(metrics=seg.metrics, holes=len(outline.holes),
                      warnings=seg.stages['fusion']['warnings']), indent=2))
# All accepted_growth_mask pixels are uncertain. Pass seg.uncertain_mask
# separately to downstream pick checks, rather than declaring inferred edges safe.
