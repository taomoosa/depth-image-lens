# Bridge-first anchor candidates

`detect_auto()` is a small shape heuristic in the same class as the backward-compatible manual `detect()`. It needs no hand ROIs, intrinsics or world calibration, but a suitable coarse upper band and optional arm exclusion remain caller responsibilities.

```python
from depth_image_lens import BridgeDrapeDetector

result = BridgeDrapeDetector().detect_auto(
    depth, difference,
    upper_band=(8, 20), min_span_px=30,
    max_height_delta_px=12, depth_jump_m=0.10,
    difference_sign="background-current",
    depth_unit="m", difference_unit="m",
)
print(result.to_dict())
```

After installation, run from the repository root:

```bash
python examples/example_auto.py
python examples/example_auto.py depth.npy difference.npy examples/config.auto.json
python -m unittest discover -s tests -t . -v
python -m tests.compare_auto
```

## Flow

1. Extract valid positive-difference foreground; exclude arms and unavailable regions.
2. Search connected components below the upper band. Prefer separated upper endpoints at similar heights with a continuous path below them. Deep U-shaped paths do not need to cross one horizontal scanline.
3. Require a bridge candidate to cover the lateral extent of other upper foreground, so half of a broken bridge does not invent an inner anchor. Inspect at most four components and two bridge candidates. A large adjacent depth discontinuity causes abstention.
4. If no bridge is found, search long hanging components for upper necks. Two separate necks, or one neck plus observed opposite foreground at similar height, can supply a candidate pair.
5. If the opposite side is unobserved, return only `hand_roi_candidates={"observed": ...}` and an `opposite_search_region`. Do not fabricate an opposite hand or its exact location. Bridge/drape remain unknown; observed extent is retained.

Upper image clipping, missing depth, inappropriate height differences or unsupported candidates can cause abstention. Candidates are not verified contacts, and no semantic hand model is used.

## Parameters and coordinates

`upper_band=(v_min,v_max)` uses coordinates projected along normalized `down`; with `(0,1)` these are image y coordinates. Candidate components are searched **below v_max**, so place that boundary near the hand-to-object connection. The default is the top quarter of the image; it may be unsuitable. A broad band does not automatically select a meaningful boundary.

`min_span_px` sets minimum lateral separation. `max_height_delta_px` allows approximate rather than exact equal image heights. `depth_jump_m` limits adjacent bridge depth jumps or neck depth spread. Tune units/invalid data first, then upper band/exclusion, separation/height allowance, and depth continuity. Example thresholds are synthetic settings, not validated universal constants.

`result.auto` records the stage, candidate boxes, projected points and opposite search band. Left/right refer to the image axis perpendicular to `down`, not physical robot IDs. `physical_contact_verified` stays false.

The independent bridge/sag/drape outputs are preserved. `image_left_local_hanging_extent_px` and `image_right_local_hanging_extent_px` add local asymmetry measurements near the candidates. The existing `drape=both` can describe a single connected bridge with a long tail on one side; it is not a symmetry assertion.

## Fixed development comparison

| Method | Matching bridge labels | False bridge | True bridge reported as no | Unknown |
|---|---:|---:|---:|---:|
| Manual ROIs | 15 | 1 | 0 | 2 |
| Automatic candidates | 9 | 1 | 0 | 8 |

These are 18 small 100x120 synthetic scenes. Unknown includes true bridges, so zero explicit false-negative `no` results does not mean full detection coverage. Manual mode received scene-specific anchor ROIs; automatic mode received only the upper band 8..20, plus an arm mask in the explicit exclusion case. Inputs are not equivalent.

Flat bridges, deep U shapes, asymmetric tails and permitted height differences can produce bridge=yes without hand ROIs. One-sided drapes with observed opposite foreground produce left/right. Separate drapes produce bridge=no and drape=both. Missing opposite foreground, upper clipping, excessive height differences, missing depth, large gaps and depth steps produce unknown. Connections only through the upper arm region produce bridge=no.

For seven scenes with defined synthetic anchor points, the 14 candidate points had mean error about 2.85 pixels and maximum about 4.93 pixels. This is not ROI IoU or real hand-detection accuracy.

The remaining false bridge is an arm slab with the same depth shape as the flat deformable-object bridge. Both manual and automatic modes return yes. An explicit arm mask suppresses yes and returns unknown; excluded pixels do not become known empty space. The ambiguity is included in the totals.

An initial run incorrectly treated half of a large broken bridge as a smaller bridge. The lateral-extent guard was added and rechecked on these same development scenes. This is not held-out evaluation or a parameter sweep. Nineteen regression tests include the known semantic failure, not nineteen correct observations.

No matched real empty-background pair has been evaluated. Real accuracy, contact recognition and physical tension remain unvalidated.
