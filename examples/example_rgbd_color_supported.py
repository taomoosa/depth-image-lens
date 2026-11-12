"""Depth-supported local color growth with gray rims and unsupported shadows.

Synthetic API demonstration, not a real-camera accuracy benchmark.
"""
import argparse
import json
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--debug', help='Optional new PNG path')
    args = parser.parse_args()
    empty_color = np.full((64, 100, 3), 140, np.uint8)
    empty_depth = np.full((64, 100), 1000, np.uint16)
    color, depth = empty_color.copy(), empty_depth.copy()
    target = np.zeros(depth.shape, bool)
    target[12:44, 10:42] = True
    color[target] = 120  # Weak luminance difference, no chromatic seed.
    depth[15:41, 13:39] = 980  # Outer three pixels have unreliable desk-like Z.
    shadow = np.zeros(depth.shape, bool)
    shadow[12:44, 42:70] = True
    color[shadow] = (20, 50, 30)  # Touching, but unlike the target's color.
    depth[22:25, 22:25] = 0  # Unknown, never filled.
    summaries = {}
    for mode in ('depth_required', 'color_seeded', 'depth_anchored', 'color_supported'):
        result = RGBDBackgroundSegmenter(fusion_mode=mode).detect(
            color, depth, background_color=empty_color, background_depth=empty_depth,
            color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
            diagnostics=True)
        summaries[mode] = dict(object_pixels=int((result.object_mask & target).sum()),
                               shadow_pixels=int((result.object_mask & shadow).sum()),
                               uncertain_pixels=int(result.uncertain_mask.sum()))
        if mode == 'color_supported':
            assert result.object_mask[target & (depth > 0)].all()
            assert not result.object_mask[shadow].any()
            assert not result.object_mask[depth == 0].any()
            assert result.unknown_mask[depth == 0].all()
            if args.debug:
                result.save_debug(args.debug, color)
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
