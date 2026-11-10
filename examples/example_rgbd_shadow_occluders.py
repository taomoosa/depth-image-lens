"""Conservative visible-object selection using existing APIs only.

Synthetic input. No claim of physical identity, hidden contour recovery or field
accuracy. All depths here are millimetres (including floating-point arrays).
"""
import json
import numpy as np
from depth_image_lens import (RGBDBackgroundSegmenter, OuterContourDetector,
                              background_difference)


empty_color = np.empty((64, 96, 3), np.uint8)
empty_color[:] = (80, 160, 100)
empty_depth = np.full((64, 96), 1000, np.uint16)
color, depth = empty_color.copy(), empty_depth.copy()
color[20:48, 20:60] = (170, 50, 120)
depth[20:48, 20:60] = 980  # Raised target: 20 mm empty-current camera-Z difference.
color[25:44, 60:83] = (8, 16, 10)  # Very dark colored table shadow, same depth.
color[14:34, 50:70] = (35, 175, 205)
depth[14:34, 50:70] = 550  # A nearer object partly occludes the target.
color[36:42, 30:38] = (35, 175, 205)
depth[36:42, 30:38] = 0  # Missing depth, not proof of either object or background.

# Keep raw pair availability separate from the selection domain. No uint16
# subtraction, zero-filling, inpainting or replacement of depth observations.
invalid_values = (65535,)
difference_mm, pair_valid = background_difference(
    depth, empty_depth, depth_unit='mm', background_unit='mm',
    invalid_values=invalid_values)
# 80 mm is a DECLARED application bound, not an estimated object height.
# It must exceed legitimate target variation and remain below unwanted near Z.
near_excluded = pair_valid & (difference_mm > 80.)
work_roi = np.ones(depth.shape, bool) & ~near_excluded
kwargs = dict(background_color=empty_color, background_depth=empty_depth,
              color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
              invalid_values=invalid_values, work_roi=work_roi, diagnostics=True)

# This is depth ONLY, not RGB AND depth. Same-table-depth shadows cannot satisfy
# the 5 mm lower threshold, regardless of their color or darkness.
depth_result = RGBDBackgroundSegmenter(
    fusion_mode='depth_required', missing_depth_policy='unknown',
    depth_threshold_m=.005).detect(color, depth, **kwargs)
object_mask = depth_result.object_mask
observed = depth_result.outline_observed_mask
uncertain = depth_result.uncertain_mask

# Optional: retain the earlier color-guided constraint. This may lose same-color
# targets. Never OR its result back into the depth gate: that resurrects shadows.
seeded = RGBDBackgroundSegmenter(
    fusion_mode='color_seeded', missing_depth_policy='unknown').detect(
        color, depth, **kwargs)
color_and_depth_mask = object_mask & seeded.object_mask
disagreement = object_mask ^ seeded.object_mask
combined_observed = observed & seeded.outline_observed_mask & ~disagreement
combined_uncertain = uncertain | seeded.uncertain_mask | disagreement

outline = OuterContourDetector().detect(object_mask, observed_mask=observed)
# An enclosed unknown patch may appear in outline.holes. It is NOT evidence of a
# physical hole. Pass uncertainty separately to downstream pick checks.
print(json.dumps(dict(
    metrics=depth_result.metrics,
    near_excluded_pixels=int(near_excluded.sum()),
    missing_pair_pixels=int((~pair_valid).sum()),
    optional_color_and_depth_pixels=int(color_and_depth_mask.sum()),
    clipped_contours=sum(c['clipped'] for c in outline.contours),
    binary_holes_not_physical_holes=len(outline.holes),
    limitations=[
        'thin_or_missing_depth_target_pixels_are_not_recovered',
        'near_exclusion_does_not_reconstruct_occluded_target',
        'coherent_depth_bias_and_same_band_occluders_can_pass',
        'camera_z_difference_is_not_table_normal_height',
        'synthetic_example_not_real_image_accuracy'],
), indent=2))
# Inspect independently: near_excluded, ~pair_valid, work_roi,
# depth_result.diagnostics['depth_difference_m'], object_mask, observed, uncertain.
# depth_result.depth_valid_mask already includes pair validity AND work_roi.
# Optional local debug output, refusing overwrite:
# depth_result.save_debug('rgbd-visible-band-debug.png', color)
