"""RGB-D segmentation to one-contour pick geometry; synthetic, not success evidence."""
import argparse
import json
import numpy as np
from depth_image_lens import (RGBDBackgroundSegmenter, OuterContourDetector,
                              OverheadPickCandidateDetector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional new debug PNG; parent must exist')
    args = parser.parse_args()
    shape = (100, 140)
    empty_color = np.full((*shape, 3), 140, np.uint8)
    empty_depth = np.full(shape, 1000, np.uint16)
    color, depth = empty_color.copy(), empty_depth.copy()
    mask = np.zeros(shape, bool)
    mask[10:90, 10:105] = True
    mask[30:58, 55:82] = False  # Hole: actual background remains visible.
    color[mask] = (35, 170, 90); depth[mask] = 980
    depth[30:60, 23:47] = 940  # Broad supported raised patch.
    depth[75, 85] = 300       # Isolated nearer spike: not a candidate.
    # Separate object inside the hole must not enter the selected component.
    color[38:50, 63:75] = (40, 80, 210); depth[38:50, 63:75] = 500
    seg = RGBDBackgroundSegmenter().detect(
        color, depth, background_color=empty_color, background_depth=empty_depth,
        color_order='RGB', registered_to_depth=True, depth_scale_m=.001)
    outline = OuterContourDetector().detect(seg.object_mask, observed_mask=seg.observed_mask)
    chosen_id = outline.largest_contour()['id']
    result = OverheadPickCandidateDetector().detect(
        depth, outline, contour_id=chosen_id, background_depth=empty_depth,
        depth_scale_m=.001, registered_to_mask=True,
        depth_valid_mask=seg.depth_valid_mask,
        segmentation_uncertain_mask=seg.uncertain_mask, diagnostics=True)
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    # result.candidates: one representative point and metrics per accepted region.
    # result.point_xy: best-ranked point, or None when no supported candidate exists.
    # result.candidate_mask: union of accepted regions within only this contour.
    if args.output:
        result.save_debug(args.output, color, color_order='RGB')


if __name__ == '__main__':
    main()
