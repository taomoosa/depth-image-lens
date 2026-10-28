"""Optional empty-reference workflow on a synthetic pair, not real accuracy evidence."""
import argparse
import json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional new debug PNG; parent must exist")
    args = parser.parse_args()
    empty = np.full((120, 160, 3), 90, np.uint8)
    color = empty.copy()
    color[20:100, 30:130] = (30, 150, 200)
    color[35:85, 78:81] = 25
    # A modest illumination change affects the empty reference only.
    reference = (empty.astype(np.int16) + 12).clip(0, 255).astype(np.uint8)
    check = np.zeros(color.shape[:2], bool)
    check[2:15, 2:25] = True  # known unchanged table, not an object sample
    depth = np.full(color.shape[:2], 1800, np.uint16)
    depth[20:100, 30:130] = 1500
    model = OverheadBoundaryDetector(background_compensate=True)
    result = model.detect(color, depth, color_order="RGB", registered_to_depth=True,
                          background_color=reference, background_check_mask=check,
                          diagnostics=True)
    print(json.dumps(result.to_dict(), allow_nan=False))
    if args.output:
        result.save_debug(args.output, color, color_order="RGB")


if __name__ == "__main__":
    main()
