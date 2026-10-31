"""Select a pixel using only vertical foreground-run clearance in one column.

Run from the repository root: python examples/example_vertical_interior.py
Add --from-boundary-result or --largest-contour for optional OpenCV examples.
These example helpers are not installed library APIs.
"""
import argparse
import json
import numpy as np


def select_vertical_interior(mask, x):
    """Return a JSON-safe selection or None when column x has no foreground.

    mask must be bool HxW; x is an integer column. Runs use inclusive foreground
    pixel-center endpoints. Clearance is min(y-start, end-y), so an endpoint or
    singleton has clearance zero (not distance to the first background pixel).
    Choose the longest run, then the topmost equal-length run, then its upper
    middle pixel for even lengths. Other columns never affect this selection.
    Image-top/bottom contact flags truncation, not a confirmed physical edge.
    Internal artificial ROI cuts cannot be identified without caller context.
    """
    a = np.asarray(mask)
    if a.dtype != bool or a.ndim != 2 or min(a.shape) < 1:
        raise ValueError('mask must be a nonempty bool HxW array')
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError('x must be an integer column')
    x = int(x)
    if not 0 <= x < a.shape[1]:
        raise ValueError('x is outside the mask width')
    changes = np.diff(np.r_[False, a[:, x], False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    if not len(starts):
        return None
    lengths = ends - starts + 1
    index = int(np.argmax(lengths))  # First (topmost) longest run.
    start, end = int(starts[index]), int(ends[index])
    y = (start + end) // 2
    top, bottom = start == 0, end == a.shape[0] - 1
    return dict(point_xy=[x, y], run_y_inclusive=[start, end],
                run_length_px=end-start+1, vertical_clearance_px=min(y-start, end-y),
                distance_to_top_end_px=y-start, distance_to_bottom_end_px=end-y,
                touches_top=top, touches_bottom=bottom, truncated=top or bottom,
                runs_y_inclusive=np.column_stack((starts, ends)).tolist())


def contour_component_mask(result, contour_id=None):
    """Compatibility wrapper for OuterContourResult.component_mask().

    The installed method retains holes and selects the visible 8-connected
    component by contour ID (largest polygon area when omitted).
    """
    return result.component_mask(contour_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x', type=int, default=16)
    parser.add_argument('--from-boundary-result', action='store_true')
    parser.add_argument('--largest-contour', action='store_true')
    args = parser.parse_args()
    mask = np.zeros((24, 32), dtype=bool)
    mask[2:9, 4:28] = True
    mask[12:22, 8:24] = True
    if args.from_boundary_result:
        from depth_image_lens import OverheadBoundaryDetector
        color = np.full((*mask.shape, 3), 200, dtype=np.uint8)
        boundary = OverheadBoundaryDetector().detect(
            color, color_order='RGB', object_mask=mask)
        mask = boundary.object_mask
    if args.largest_contour:
        from depth_image_lens import OuterContourDetector
        outline = OuterContourDetector().detect(mask)
        mask = contour_component_mask(outline)
    print(json.dumps(select_vertical_interior(mask, args.x), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
