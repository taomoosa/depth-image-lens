# Optional mask cleanup

`cleanup_mask()` is a separate, NumPy-only postprocessing step for an existing
boolean segmentation mask. It can fill small enclosed holes, optionally close
narrow open notches, and smooth a jagged exterior without changing the color or depth segmentation thresholds. No
operation is enabled by default, and no existing detector calls it implicitly.

Start with **hole filling alone**. Enable smoothing only if its changes are
acceptable for your object's corners and thin features. A better-looking mask is
not additional sensor evidence or proof of better segmentation.

## Existing functions and this addition

- `OuterContourDetector(smoothing_px=...)` smooths contour samples used for turn
  and section analysis. It does **not** replace its input mask or raw geometry.
- The boundary detector's grayscale closing measures local darkness. It is not
  binary-mask hole filling.
- RGB-D `color_seeded` growth uses color/depth evidence and seed/growth settings.
  It is not an independent shape-only cleanup operation.
- `cleanup_mask()` explicitly returns a repaired mask hypothesis, its original
  mask, and every changed pixel. It does not modify segmenter parameters,
  thresholds, source evidence, depth values or existing result objects.

## Independent use

```python
from depth_image_lens import cleanup_mask

# object_mask: HxW bool ndarray; input is never modified.
repair = cleanup_mask(object_mask, max_hole_area=9)
repaired_mask = repair.object_mask
original_mask = repair.original_mask

# Independently enable one small exterior-smoothing pass if appropriate.
repair = cleanup_mask(
    object_mask,
    max_hole_area=9,
    smoothing_radius_px=1,
    observed_mask=observed_mask,
    protected_mask=exclusion_or_deferred_mask,
)
print(repair.to_dict())
print(repair.added_mask.sum(), repair.removed_mask.sum())
```

All arrays must be boolean on the same pixel grid. Numeric 0/255 masks must be
converted explicitly by the caller. Noncontiguous views are accepted. Empty
foreground is supported, but zero-height/zero-width arrays are rejected. Inputs
and output snapshots have separate buffers; result arrays themselves are ordinary
mutable NumPy arrays, not a read-only security boundary.

## Settings and exact behavior

| Setting | Default | Meaning |
|---|---:|---|
| `max_hole_area` | `0` | Nonnegative Python integer. Zero disables filling. Fill only enclosed background components with pixel area **at most** this limit. |
| `smoothing_radius_px` | `0` | Python integer `0..5`. Zero disables smoothing. Radius 1 uses a 3×3 square majority window; radius 2 uses 5×5. One pass only. |
| `closing_radius_px` | `0` | Python integer `0..5`. Square closing radius: 1 proposes a 3×3 closing. Requires positive `max_gap_area`. |
| `max_gap_area` | `0` | Nonnegative Python integer. Inclusive cap on each connected **addition proposal**, not on the entire exterior background or a newly enclosed pocket. Zero disables closing. |
| `observed_mask` | All true | False means unavailable, not confidently background. Preserve its input values and do not use its neighborhood to drive corrections. |
| `protected_mask` | All false | Preserve both foreground and background values. Use for explicit exclusions, unresolved decisions, or known features that must stay unchanged. |

Pixel areas scale quadratically with resolution. Radii are pixels, not metres.
There is no resolution-independent recommended hole limit. `9` and `1` above are
small starting examples, not validated camera defaults.

### Hole filling

1. Label background with **8-connectivity**, including diagonal contact.
2. Reject anything connected to the image frame, even by a diagonal path.
3. Reject the entire hole if any pixel in it or its one-pixel rim is unobserved
   or protected. Do not fill the known part around an unknown interior.
4. Enforce the inclusive area limit and require exactly one neighboring
   8-connected foreground component. A ring containing an independent island
   cannot be filled into a single merged object.
5. Add accepted pixels without changing the exterior boundary.

A cropped object can still have a fully enclosed, observed hole filled away from
the frame. A notch or channel open to the exterior is not a hole and is not filled.

### Narrow gaps connected to the exterior

An exterior-connected slit is not an enclosed hole. Increasing `max_hole_area`
cannot fill it, and topology-preserving smoothing deliberately keeps an open
route. Use the separate, disabled-by-default closing operation when a small
open slit is known to be segmentation damage:

```python
repair = cleanup_mask(
    segmented.object_mask,
    closing_radius_px=1,
    max_gap_area=64,
    observed_mask=segmented.outline_observed_mask,
    protected_mask=segmented.suppressed_mask | segmented.deferred_mask,
)
mask = repair.object_mask
uncertain = segmented.uncertain_mask | repair.changed_mask
# repair.closed_gap_mask identifies the pixels added by gap closing.
```

Start with radius 1 and an area cap appropriate to the defect. For a straight,
axis-aligned slit, radius 1 can fill widths 1–2 pixels, radius 2 widths 3–4, and
radius 3 widths 5–6; these are synthetic geometry examples, not universal width
guarantees for diagonal or curved shapes. A 2×23 slit needs `max_gap_area >= 46`.
Increase the radius only if the missing width requires it; raise the area cap
only to cover the intended missing area. Large radii can erase real concavities.

Processing order is enclosed-hole filling, gap closing, then smoothing:

1. Compute square binary closing (dilation then erosion), but propose additions
   only. Closing never removes supplied foreground.
2. Label proposed additions with 8-connectivity. Reject an entire addition
   component if it exceeds the inclusive area cap, touches more than one current
   foreground component, or lacks a foreground neighbor.
3. Reject the complete proposal if it intersects a `2 * closing_radius_px` frame
   band or unknown/protected halo. This covers the full closing footprint; one
   blocked pixel may prevent repair of the whole slit. Do not clip a candidate
   into a misleading partial repair.
4. Preserve original enclosed holes and their one-pixel rim. Closing is separate
   from `max_hole_area` and does not fill retained enclosed holes.
5. Preserve accepted gap additions and their one-pixel rim during smoothing.

Distinct foreground components cannot be joined. **Background topology can
change intentionally**: a narrow mouth into a large chamber may be sealed while
leaving the chamber as a new enclosed hole. `max_gap_area` bounds added pixels,
not the chamber's area. Newly enclosed pockets are not filled in this call;
repeated cleanup calls could fill them later, so apply once to the original mask.

A real narrow notch is indistinguishable from missing segmentation. Closing may
incorrectly seal a real opening between two parts of the same object. Protect
known openings explicitly. Cropped gaps and gaps over unknown or excluded data
are deliberately left unchanged. Omitting protection is not evidence that those
pixels are observed; it can produce invented geometry. No depth is reconstructed.

`examples/example_mask_cleanup_gaps.py` demonstrates nine cases. Its 46-pixel
slit is completely restored with radius 1 / area 64, unchanged by hole filling
alone, rejected at area 32, and rejected when unknown/protected. A real slit of
the same shape produces 46 incorrect additions. Real-image accuracy is untested.

### Exterior smoothing

After optional filling and closing, propose one square-window majority pass. Freeze all
original enclosed holes and their one-pixel foreground rim, including holes just
filled. Also freeze a radius-wide image-frame band and every pixel whose window
intersects unknown/protected data.

Apply proposed changes sequentially in row-major order. A local 3×3 connectivity
check rejects a change that could create, erase, split or join an 8-connected
foreground or background component. This keeps neighboring mask components
separate, preserves thin connecting necks, and leaves an 8-connected route through
open gaps unless the earlier closing operation was explicitly enabled. Smoothing
is deliberately more conservative than ordinary median filtering,
opening or closing.

These guarantees refer to **8-connected foreground and 8-connected background**,
not the conventional dual 8/4 digital-topology definition. They do not guarantee
unchanged gap width, corner position, object area or semantic identity. The pass
is deterministic but direction-biased, not generally rotation-invariant or
idempotent. Repeated calls can keep moving details; apply once to the original
segmentation rather than repeatedly cleaning previous outputs. Highly noisy
masks have many sequential checks; no real-time performance guarantee is made.

## Output and uncertainty

`MaskCleanupResult` contains:

- `original_mask`: a copy of the supplied mask.
- `object_mask`: the repaired hypothesis.
- `added_mask`, `removed_mask`, and property `changed_mask`: exact differences
  against the original, including removals that might otherwise be overlooked.
- `filled_hole_mask`: additions caused by filling.
- `closed_gap_mask`: additions caused by bounded open-gap closing (always a bool
  array from `cleanup_mask`, including when disabled).
- `smoothed_added_mask`, `smoothed_removed_mask`: changes caused by smoothing.
- `observed_mask`, `protected_mask`: unchanged copies of the caller's masks.
- `parameters`, `metrics`, `warnings`, and JSON-compatible `to_dict()`.

`added_mask == filled_hole_mask | closed_gap_mask | smoothed_added_mask` and
`removed_mask == smoothed_removed_mask`. Filled and gap-closed pixels cannot subsequently be
removed by smoothing. Metrics count proposed smoothing changes and topology
rejections. `enclosed_holes` is `None` when all operations are disabled, because
no hole analysis was run.

Corrections remain **inferred geometry**, even if the original pixel had a valid
color/depth observation. Do not mark them as new depth support, clear uncertainty,
or overwrite `deferred_mask`, `suppressed_mask`, source diagnostics or fusion
decisions. This function does not infer unknown/deferred semantics from a binary
mask; the caller must provide those masks.

## RGB-D segmentation and outlines

```python
from depth_image_lens import cleanup_mask, OuterContourDetector

# segmented = RGBDBackgroundSegmenter(...).detect(...)
protected = segmented.deferred_mask | segmented.suppressed_mask
repair = cleanup_mask(
    segmented.object_mask,
    max_hole_area=9,
    smoothing_radius_px=0,
    observed_mask=segmented.outline_observed_mask,
    protected_mask=protected,
)
combined_uncertain = segmented.uncertain_mask | repair.changed_mask

# Geometry for display or a separate, explicitly inferred candidate.
outline = OuterContourDetector().detect(
    repair.object_mask,
    observed_mask=segmented.outline_observed_mask,
)
# Keep repair.changed_mask and combined_uncertain alongside this outline.
# The outline API does not ingest an uncertainty mask.
```

`RGBDBackgroundResult.outline_observed_mask` already excludes suppressed and
deferred pixels, so passing `protected` as well is explicit but redundant for
those pixels. Also OR any application-specific excluded regions into `protected`.
For a stricter policy, additionally protect `segmented.uncertain_mask`; this can
prevent all corrections on color-only candidates. That is expected, not an error.
The default example permits geometric inference on available uncertain pixels
while retaining all their uncertainty.

Use `segmented.object_mask` and the original depth/validity/uncertainty evidence
for evidence-only measurement or safety-sensitive pick/grasp decisions. An
outline computed from the repaired mask describes that hypothesis; it does not
verify a material edge. Do not treat a cleanup result as permission for a robot
motion. Other detectors with an explicit `object_mask` input can likewise consume
the repaired hypothesis, but their result is a new analysis and must retain the
cleanup provenance separately.

Run a complete synthetic RGB-D → cleanup → outline example and comparisons:

```bash
python examples/example_mask_cleanup.py
python examples/example_mask_cleanup_gaps.py
python -m unittest tests.test_mask_cleanup -v
python -m unittest discover -s tests -t . -v
```

The RGB-D/outline example needs the existing OpenCV extras; `cleanup_mask()` itself
does not. The example prints strict JSON and writes no files. Its thin object has
only 2 mm of depth difference, below the unchanged 5 mm depth threshold; brightness
supplies the initial mask, then cleanup fills a four-pixel segmentation defect.

## Reproducible synthetic comparisons

The example includes 15 deterministic 64×80 cases with known, invented ground
truth. IoU is intersection / union, with two empty masks defined as IoU 1. Error
pixels count the symmetric difference from ground truth. It also reports added
and removed pixels, incorrect additions, lost true foreground, and component
counts. Rounded results from the included example:

| Case | Raw IoU | Repaired IoU | Error pixels before → after |
|---|---:|---:|---:|
| Small segmentation holes, filling only | 0.994196 | 1.000000 | 13 → 0 |
| Jagged exterior, radius 1 | 0.990227 | 0.998214 | 22 → 4 |
| Holes and jagged exterior | 0.984451 | 0.998214 | 35 → 4 |
| Real small hole, indistinguishable from damage | 1.000000 | 0.998214 | 0 → 4 |
| Large real hole retained, exterior corners rounded | 1.000000 | 0.997980 | 0 → 4 |
| Neighboring objects, radius 2, still two components | 1.000000 | 0.973588 | 0 → 58 |
| Thin appendage can shorten | 1.000000 | 0.996889 | 0 → 7 |
| Sharp corners can round | 1.000000 | 0.998214 | 0 → 4 |
| Protected corners | 1.000000 | 1.000000 | 0 → 0 |
| Cropped object, frame unchanged | 1.000000 | 0.999313 | 0 → 2 |
| Open gap cannot be reconstructed | 0.969196 | 0.967411 | 69 → 73 |
| Unknown four-pixel hole protected, other hole filled | 0.994196 | 0.998214 | 13 → 4 |
| Deferred nine-pixel hole protected, other hole filled | 0.994196 | 0.995982 | 13 → 9 |
| Completely missing object stays missing | 0.000000 | 0.000000 | 2240 → 2240 |
| Empty foreground stays empty | 1.000000 | 1.000000 | 0 → 0 |

These results intentionally include regressions. An intact mask does not need
smoothing. Connectivity preservation does not mean pixel accuracy, and a visually
clean mask can be less correct. Unit tests also cover diagonal leaks, holes with
independent islands, unknown/protected hole rims, thin necks, isolated pixels,
full and tiny masks, strict validation, noncontiguous input, nonmutation,
independent component-count checks, and operation without OpenCV.

**Real-data accuracy has not been validated.** No binary-mask-only method can
distinguish a true small opening from a missing segmentation patch, recover a
fully missing object, infer occluded/cropped material, or safely separate objects
already merged by segmentation. Keep the original, inspect changed pixels, and
validate on representative captures before relying on the repaired geometry.
