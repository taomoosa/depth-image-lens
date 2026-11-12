"""Compare whole-region support with local seeds on sparse synthetic depth."""
import json
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter


def main():
    background_color = np.full((80, 120, 3), 140, np.uint8)
    color = background_color.copy()
    color[20:60, 20:60] = 120
    background_depth = np.full((80, 120), 1000, np.uint16)
    depth = background_depth.copy()
    yy, xx = np.indices(depth.shape)
    target = color[..., 0] == 120
    depth[target & ((yy + xx) % 2 == 0)] = 986
    report = []
    for mode in ('color_supported', 'color_region'):
        result = RGBDBackgroundSegmenter(fusion_mode=mode).detect(
            color, depth, background_color=background_color,
            background_depth=background_depth, color_order='RGB',
            registered_to_depth=True, depth_scale_m=.001, diagnostics=True)
        report.append(dict(mode=mode, selected=int(result.object_mask.sum()),
                           target=int(target.sum()),
                           regions=result.stages['fusion'].get('regions', [])))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
