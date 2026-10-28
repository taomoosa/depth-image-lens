"""Small in-memory shadow example; analytic colors are not field validation."""
import json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector

empty = np.full((100,140,3), (110,160,190), np.uint8)
current = empty.copy()
current[25:80,60:120] = (185,80,60)
current[40:90,10:50] = np.rint(current[40:90,10:50]*0.5).astype(np.uint8)
check = np.zeros(current.shape[:2], bool); check[5:15,10:30] = True
seed = np.zeros_like(check); seed[30:40,70:80] = True
model = OverheadBoundaryDetector(background_shadow_mode="uniform_color")
result = model.detect(current, color_order="RGB", background_color=empty,
                      background_check_mask=check, object_color_seed=seed,
                      diagnostics=True)
print(json.dumps(result.stages["shadow_separation"], allow_nan=False, indent=2))
debug = result.render_debug(current)  # uint8 RGB; no implicit filesystem writes
# result.save_debug("shadow-debug.png", current)  # explicit, non-overwriting
assert result.object_mask[30:40,70:80].all()
assert not result.object_mask[40:90,10:50].any()
