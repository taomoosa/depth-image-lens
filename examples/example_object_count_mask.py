"""Direct masks and cleanup provenance, without segmentation, files, or robot I/O."""
import json
import numpy as np
from depth_image_lens import OverheadObjectCounter, cleanup_mask


def run_examples():
    shape = (60, 80)
    color = np.full((*shape, 3), 150, np.uint8)
    depth = np.full(shape, 1000, np.uint16)
    empty_depth = depth.copy()
    mask = np.zeros(shape, bool)
    mask[10:40, 10:35] = True
    color[10:40, 10:35] = (210, 35, 35)
    mask[10:40, 35:60] = True  # Touching colors at the same depth.
    color[10:40, 35:60] = (35, 35, 210)
    depth[mask] = 980
    counter = OverheadObjectCounter()
    depth_inputs = dict(depth=depth, background_depth=empty_depth,
                        depth_scale_m=.001, registered_to_depth=True)
    examples = {
        'external_mask_with_depth': counter.detect_mask(
            color, mask, color_order='RGB', **depth_inputs).to_dict(),
        'mask_only': counter.detect_mask(color, mask, color_order='RGB').to_dict(),
    }
    raw = mask.copy()
    raw[20:22, 20:22] = False
    repair = cleanup_mask(raw, max_hole_area=4)
    examples['cleaned_mask_with_provenance'] = counter.detect_mask(
        color, repair.object_mask, color_order='RGB', **depth_inputs,
        cleanup_changed_mask=repair.changed_mask,
    ).to_dict()
    return examples


if __name__ == '__main__':
    print(json.dumps(run_examples(), indent=2, allow_nan=False))
