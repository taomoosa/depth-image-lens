"""Depth-supported gray-object refinement with explicit inference and uncertainty.

Run from an installed checkout: python examples/example_rgbd_anchored.py
Optional: --debug /tmp/anchored.png writes a diagnostic RGB image.
This generated scene is an API example, not a sensor-accuracy benchmark.
"""
import argparse
import json
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug', help='Optional output PNG path')
    args = parser.parse_args()
    empty_color = np.full((64, 96, 3), 140, np.uint8)
    empty_depth = np.full((64, 96), 1000, np.uint16)
    color, depth = empty_color.copy(), empty_depth.copy()
    target = np.zeros(depth.shape, bool)
    target[14:49, 18:65] = True
    target[28:35, 34:41] = False  # Observed background hole, no closing.
    target[30, 65:82] = True  # One-pixel appendage, no eroded-core requirement.
    color[target] = 40
    depth[target] = 980
    color[50:56, 18:65] = 30  # A dark shadow has no supporting foreground depth.
    depth[20:23, 22:26] = 0  # Keep missing sensor support unknown.

    result = RGBDBackgroundSegmenter(fusion_mode='depth_anchored').detect(
        color, depth, background_color=empty_color, background_depth=empty_depth,
        color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
        diagnostics=True)
    outline = OuterContourDetector().detect(
        result.object_mask, observed_mask=result.outline_observed_mask)
    assert not result.object_mask[depth == 0].any()
    assert result.unknown_mask[depth == 0].all()
    assert result.unknown_mask[result.inferred_foreground_mask].all()
    assert result.object_mask[30, 65:82].all()
    print(json.dumps(dict(metrics=result.metrics,
                         contour_count=len(outline.contours),
                         warnings=result.stages['fusion']['warnings']), indent=2))
    if args.debug:
        result.save_debug(args.debug, color)
    # Pass result.uncertain_mask separately to downstream grasp checks.
    # A "confident" rule category does not certify a surface or a safe grasp.


if __name__ == '__main__':
    main()
