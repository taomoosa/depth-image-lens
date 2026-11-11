"""Self-contained overhead visible-color counting examples; no files or robot I/O."""
import json
import numpy as np
from depth_image_lens import OverheadObjectCounter


def scene():
    background_color = np.full((80, 100, 3), 150, np.uint8)
    background_depth = np.full((80, 100), 1000, np.uint16)
    return background_color.copy(), background_depth.copy(), background_color, background_depth


def run_examples():
    def run(arrays, **options):
        color, depth, empty_color, empty_depth = arrays
        return OverheadObjectCounter(max_reference_separation_m=.15).detect(
            color, depth, background_color=empty_color, background_depth=empty_depth,
            color_order='RGB', depth_scale_m=.001, registered_to_depth=True, **options).to_dict()

    examples = {'empty': run(scene())}
    arrays = color, depth, _, _ = scene()
    color[20:60, :85] = (210, 35, 35)  # Red object is cropped at the left frame.
    color[20:60, :30] = (105, 18, 18)  # Approximate proportional shading.
    depth[20:60, :85] = 980
    color[10:70, 35:60] = (35, 35, 210)  # Blue object splits visible red into two parts.
    depth[10:70, 35:60] = 960
    examples['overlap_shading_crop'] = run(arrays)

    arrays = color, depth, _, _ = scene()
    color[10:35, 10:35] = (210, 35, 35)
    depth[10:35, 10:35] = 980
    color[10:60, 60:80] = (200, 140, 100)
    depth[10:60, 60:80] = 700
    arm_mask = np.zeros(depth.shape, bool)
    arm_mask[11:59, 61:79] = True  # A one-pixel residual is covered by the guard.
    examples['imperfect_arm_mask'] = run(arrays, arm_mask=arm_mask)

    arrays = color, depth, _, _ = scene()
    color[10:35, 10:35] = (210, 35, 35)  # No supporting depth change.
    examples['thin_or_shadow'] = run(arrays)
    return examples


if __name__ == '__main__':
    print(json.dumps(run_examples(), indent=2, allow_nan=False))
