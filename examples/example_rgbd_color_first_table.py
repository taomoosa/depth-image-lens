"""Recover a thin synthetic region after a tabletop color and depth shift."""
import numpy as np

from depth_image_lens import RGBDBackgroundSegmenter

background_color = np.full((400, 640, 3), 140, np.uint8)
background_depth = np.full((400, 640), 1000, np.uint16)
color = np.full_like(background_color, 100)  # changed table appearance
depth = np.full_like(background_depth, 994)  # frame-wide +6 mm empty-current difference
color[110:290, 220:420] = (205, 70, 125)
depth[110:290, 220:420] = 993                # +1 mm object relative to the table

for label, anchors in (("ordinary", None),
                       ("anchored", [(35, 35), (70, 45), (570, 45), (605, 35)])):
    detector = RGBDBackgroundSegmenter(
        fusion_mode="color_first", first_table_xy=anchors,
        first_table_statistic="median",
    )
    result = detector.detect(
        color, depth, background_color=background_color,
        background_depth=background_depth, color_order="RGB",
        registered_to_depth=True, depth_scale_m=.001, diagnostics=True,
    )
    fusion = result.stages["fusion"]
    print(label, "object pixels:", int(result.object_mask.sum()))
    if anchors:
        table = fusion["table_reference"]
        print("table:", table["status"], "offset (m):", table["offset_m"])
        print("region references:", [r.get("table_reference_source") for r in fusion["regions"]])
        print("inferred foreground pixels:", int(result.inferred_foreground_mask.sum()))
