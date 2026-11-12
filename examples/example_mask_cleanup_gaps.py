"""Controlled open-gap repair examples. No files written; synthetic data only."""
import json
import numpy as np
from depth_image_lens import cleanup_mask


def run_examples():
    truth = np.zeros((64, 80), bool)
    truth[12:52, 12:68] = True
    slit = truth.copy()
    slit[12:35, 38:40] = False  # 2px wide, 23px deep; 46 missing pixels.
    wide = truth.copy()
    wide[12:35, 38:43] = False
    separated = truth.copy()
    separated[:, 38:40] = False
    unknown = np.ones_like(truth)
    unknown[20, 38] = False
    protected = np.zeros_like(truth)
    protected[12:35, 38:40] = True
    settings = dict(closing_radius_px=1, max_gap_area=64)
    cases = [
        ('hole_filling_alone', truth, slit, dict(max_hole_area=1000)),
        ('smoothing_alone', truth, slit, dict(smoothing_radius_px=1)),
        ('narrow_gap_repaired', truth, slit, settings),
        ('area_limit_rejects_whole_gap', truth, slit, dict(closing_radius_px=1, max_gap_area=32)),
        ('wide_gap_not_closed', truth, wide, settings),
        ('separate_objects_not_joined', separated, separated, settings),
        ('unknown_gap_not_repaired', truth, slit, dict(**settings, observed_mask=unknown)),
        ('protected_gap_not_repaired', truth, slit, dict(**settings, protected_mask=protected)),
        ('real_notch_indistinguishable', slit, slit, settings),
    ]
    rows = []
    for name, expected, raw, kwargs in cases:
        repaired = cleanup_mask(raw, **kwargs)
        rows.append(dict(case=name, raw_error_pixels=int((raw != expected).sum()),
                         repaired_error_pixels=int((repaired.object_mask != expected).sum()),
                         gap_added_pixels=int(repaired.closed_gap_mask.sum()),
                         false_added_pixels=int((repaired.added_mask & ~expected).sum()),
                         parameters=repaired.parameters))
    return rows


if __name__ == '__main__':
    print(json.dumps(dict(synthetic_only=True, comparisons=run_examples()),
                     indent=2, allow_nan=False))
