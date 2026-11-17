"""Write grouped full diagnostics from a synthetic registered RGB-D scene.

Run: python examples/example_full_debug.py /tmp/lens-full-debug-new
The destination must not exist. No camera, external data or GUI is required.
"""
import argparse
from pathlib import Path
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='new output directory')
    parser.add_argument('--mode', choices=('color_model', 'color_first'), default='color_model')
    args = parser.parse_args()
    background_color = np.full((400, 640, 3), (175, 170, 160), np.uint8)
    color = background_color.copy()
    background_depth = np.full((400, 640), 1000, np.uint16)
    depth = background_depth.copy()
    color[90:310, 140:490] = (45, 110, 185)
    depth[90:310, 140:490] = 990
    depth[120:130, 240:250] = 0  # Missing measurements stay distinct.
    segmenter = RGBDBackgroundSegmenter(
        fusion_mode=args.mode, model_training_policy='color_core')
    result = segmenter.detect(color, depth,
        background_color=background_color, background_depth=background_depth,
        color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
        diagnostics=True)
    index = result.save_debug_bundle(args.output, color,
        columns=6, max_rows=4, tile_width=640, models='all')
    print(index)


if __name__ == '__main__':
    main()
