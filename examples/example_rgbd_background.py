"""Two current/empty RGB-D pairs; synthetic API example, not accuracy evidence."""
import argparse
import json
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional new debug PNG path; parent must exist')
    args = parser.parse_args()
    background_color = np.full((80, 120, 3), 140, np.uint8)
    background_depth = np.full((80, 120), 1000, np.uint16)
    color, depth = background_color.copy(), background_depth.copy()
    # Same-colored nonconvex object, with an actual empty hole.
    obj = np.zeros(depth.shape, bool)
    obj[12:65, 10:65] = True
    obj[12:30, 45:65] = False
    obj[36:48, 23:36] = False
    depth[obj] = 970
    # Separate thin colored candidate below the depth threshold.
    color[25:55, 85:105] = (30, 150, 210)
    depth[25:55, 85:105] = 999
    # A small patch has missing depth; color can retain it with uncertainty.
    depth[35:40, 90:95] = 0
    # Mild brightness-only change; ignored with this luma tolerance.
    color[67:75, 15:65] = 120
    segmentation = RGBDBackgroundSegmenter(
        background_chroma_tolerance=20, background_luma_tolerance=60,
        depth_threshold_m=.005, depth_margin_m=.001,
        fusion_mode='union', missing_depth_policy='color',
    ).detect(color, depth, background_color=background_color,
             background_depth=background_depth, color_order='RGB',
             registered_to_depth=True, depth_scale_m=.001, diagnostics=True)
    outline = OuterContourDetector().detect(
        segmentation.object_mask, observed_mask=segmentation.observed_mask)
    print(json.dumps(dict(segmentation=segmentation.to_dict(),
                          contours=len(outline.contours), holes=len(outline.holes)),
                     indent=2, allow_nan=False))
    # Optional existing internal boundary analysis:
    # OverheadBoundaryDetector().detect(color, depth, color_order='RGB',
    #     registered_to_depth=True, depth_scale_m=.001,
    #     object_mask=segmentation.object_mask)
    # Numeric evidence is available as segmentation.diagnostics:
    # depth_difference_m, background_combined_score (threshold 1), etc.
    if args.output:
        segmentation.save_debug(args.output, color, color_order='RGB')


if __name__ == '__main__':
    main()
