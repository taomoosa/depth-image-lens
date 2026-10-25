"""Synthetic visible-corner example; no dataset, network or robot access."""
import json
import numpy as np
from depth_image_lens import OverheadCornerDetector


def main():
    mask = np.zeros((180, 220), dtype=bool)
    mask[40:140, 50:170] = True
    color = np.full((180, 220, 3), 70, dtype=np.uint8)
    color[mask] = [30, 80, 200]
    depth = np.where(mask, 980, 1000).astype(np.uint16)
    # Use OverheadCornerDetector() to retain the original compatibility defaults.
    detector = OverheadCornerDetector.recommended_for_tuning(
        fitted_angle_tolerance_deg=25., angle_stability_deg=10.,
        max_intersection_shift_px=2., min_fit_support=4,
        depth_step_m=0.015, boundary_mixing_m=0.015)
    flat = detector.detect(color, depth, color_order="RGB", registered_to_depth=True,
                           object_mask=mask, depth_scale_m=0.001, diagnostics=True)
    depth[40:140, 50:65] = 920  # Internal discontinuity, not the object/table boundary.
    folded = detector.detect(color, depth, color_order="RGB", registered_to_depth=True,
                             object_mask=mask, depth_scale_m=0.001)
    print(json.dumps({"scope": "synthetic only", "flat": flat.to_dict(),
                      "internal_step": folded.to_dict()}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
