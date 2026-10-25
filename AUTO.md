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
```

## Flow

1. Extract valid positive-difference foreground under the selected input contract; exclude arms and unavailable regions. In `positive_foreground` mode, finite zero is observed background/exclusion, while raw-depth or mask invalidity remains unknown.
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

## Close views: missing hands and the upper third

The stationary, suspended-from-above assumption is compatible with this detector.
It does not make an invisible connection observable. The current class tests a
visible path between image anchor candidates. Its independent feature pass adds
explicitly inferred one/two-support hypotheses from the visible silhouette, but
cannot uniquely recover hidden supports or confirm physical grasp.

- If both candidate connections and the path remain visible, hand interiors and
  entire arms need not be visible. Supply a suitable `upper_band` and an arm mask
  if needed. A positive result is still a shape candidate, not verified contact.
- If a substantial component touches an upstream image edge, automatic mode
  returns `upper_foreground_clipped`. This covers all upstream edges for the
  supplied `down`, including a tilted direction. It retains foreground/work masks
  for the conservative anchor decision. Independent contour measurements and
  support hypotheses continue with their inferred reference explicitly labeled.
- For manual mode, translate hand ROIs into the cropped coordinates and mark
  missing supports `hand_visibility=("hidden", "hidden")` (or the appropriate
  per-side values). Do not move invisible ROIs into the image to obtain an answer.
- An observed empty crop can return `bridge=no` with
  `no_foreground_in_observed_image`. This is absence in the visible image only;
  it is not evidence that no bridge exists above or outside the crop. All outputs
  retain `scope=visible_shape_heuristic_not_holding_or_tension`.

Crop depth, difference and every optional mask identically. Subtract the crop
origin from ROI coordinates; recompute the projected upper band after cropping
or translation. A pure translation does not change `down`. Bridge/drape image
geometry needs no intrinsics. If the separate metric LiftDetector is used,
update its camera principal point as described in the lift contract.

`drape=left/right` names the image transverse direction, not a physical hand ID.
A mapping supplied by the calling application is required before associating it
with robot hands, and shape alone still does not prove holding. When a support
or opposite search region is hidden, this API does not promote a visible tail
into a one-handed grasp classification. Threshold adjustment cannot recover
missing semantic or off-screen evidence.


Feature-wise continuation: `stage_results` now runs alongside the conservative
automatic-anchor result. An upper-clipped or anchorless silhouette retains
morphology, visible-contour sag measurements and explicitly inferred static
one/two-support hypotheses. Earlier automatic-anchor limitations above still
apply to observed bridge and physical support identification; they no longer
suppress these independent measurements. See RGBD.md for reference assumptions,
quality warnings and root-output semantics.
