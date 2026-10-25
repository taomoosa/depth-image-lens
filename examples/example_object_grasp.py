"""Minimal synthetic object-only inference; not real grasp validation."""
import json
import numpy as np
from depth_image_lens import ObjectOnlyGraspDetector, positive_difference_16uc1


def main():
    background = np.full((60,100), 1500, np.uint16)
    current = background.copy()
    object_mask = np.zeros(current.shape, bool)
    object_mask[10:50,20:80] = True
    current[object_mask] = 1400
    difference, valid = positive_difference_16uc1(current, background)
    result = ObjectOnlyGraspDetector().detect_16uc1(
        current, object_mask, difference=difference, background_valid=valid,
        diagnostics=True,
    )
    print(json.dumps(result.to_dict(), allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
