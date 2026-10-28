"""Seedless background comparison; illustrative colors are not field calibration."""
import json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector

empty = np.full((100,140,3), (110,160,190), np.uint8)
current = empty.copy()
current[25:80,60:120] = (185,80,60)
current[40:90,10:50] = np.rint(current[40:90,10:50]*0.5).astype(np.uint8)
check = np.zeros(current.shape[:2], bool); check[5:15,10:30] = True
model = OverheadBoundaryDetector(background_difference_mode="separate_tolerances",
                                 background_chroma_tolerance=30.,
                                 background_luma_tolerance=60.)
result = model.detect(current, color_order="RGB", background_color=empty,
                      background_check_mask=check, diagnostics=True)
print(json.dumps(result.stages["background_difference"], allow_nan=False, indent=2))
debug = result.render_debug(current)  # RGB uint8, no implicit file write
score = result.diagnostics["background_combined_score"]
assert score.shape == current.shape[:2]
# result.save_debug("tolerances.png", current)  # explicit, non-overwriting
