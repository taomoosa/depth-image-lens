# Debug images and tuning

Detection never writes files. Public spatial **results** expose `render_debug`
and explicit `save_debug`. Numerical computation uses NumPy, with OpenCV also required by Corner, Boundary and OuterContour; rendering
uses the optional existing OpenCV dependency (`pip install -e '.[debug]'`).
No camera, model, GUI or remote service is started.

See [the complete parameter reference](TUNING.md) for exact defaults, units,
threshold directions, call-time controls and runnable examples for every detector.

```python
result = detector.detect(..., diagnostics=True)
image_rgb = result.render_debug(color_rgb, color_order="RGB")
result.save_debug("debug.png", color_rgb, color_order="RGB")
# Or without RGB:
image_rgb = result.render_debug(depth=depth_uint16, depth_scale_m=.001)
# Explicit replacement only:
result.save_debug("debug.png", depth=depth_uint16, overwrite=True)
```

| Computation | Renderable result | Useful tuning view |
| --- | --- | --- |
| `BridgeDrapeDetector` | `BridgeDrapeResult` | Foreground, observed/inferred path, anchors, sag and separate features. |
| `IndependentGraspDetector` | `GraspResult` | Contact/uncertainty masks, per-side support, consistency and statuses. |
| `ObjectOnlyGraspDetector` | `ObjectOnlyGraspResult` | Eligible lifted components, width, reference/coverage, optional gripper exclusion. |
| `LiftDetector` | `LiftResult` | `diagnostics=True`: selected/valid/elevated/near-table masks and thresholds. |
| `fit_table_plane` | `PlaneFitResult` | `diagnostics=True`: table ROI, valid support and residual summary. |
| `SeededColorAssistant` | `ColorAssistResult` | Color assignments/uncertainty, seed quality and thresholds. |
| `ReferenceGripperSegmenter.calibrate` | `GripperReference` | Generated reference hands, static ROI and coarse search boxes. |
| `ReferenceGripperSegmenter.segment` | `ReferenceSegmentationResult` | Hand/object/unknown masks, per-side reference quality. |
| `DepthSurfaceDetector` | `SurfaceResult` | Independent elevation/nonflat/unknown masks and local measurements. |
| `OverheadBoundaryDetector` | `BoundaryResult` | Object domain, color/depth bands, relative islands, background distances and optional topology. |
| `OuterContourDetector` | `OuterContourResult` | Ordered straight/curve intervals, corners, clipping and section IDs; [options](OUTLINE.md). |
| `OverheadCornerDetector` | `CornerResult` | Specialized contour, candidate ID/status/reason overlay; [API](CORNER.md). |

`analyze_color_geometry` returns ordinary component results; render those
individually. `CameraIntrinsics`, `TablePlane` and numeric difference helpers have
no spatial result renderer. Render their downstream results. There is no renderer
on every trivial data container.

Common result options: `color_order="RGB"` or `"BGR"` controls both input and
output; `depth_scale_m=.001`, `mask_alpha=.35`, `panel=True`, `max_lines=32`.
Source color is uint8 HxWx3. Source depth is HxW optical-axis Z, explicit scale
for every dtype. Noncontiguous inputs are accepted; no source/result mutation.
All result masks must match source dimensions. Depth display is percentile-scaled
and invalid depth is magenta; display intensity is **not** metric height.
Independent mask tiles preserve overlapping evidence. Unknown/uncertain overlays
are magenta; exclusions orange. Titles report processing-pixel counts. Large
views are downscaled for display; the original arrays/thresholds are unchanged.
At most 13 tiles are shown; omitted-mask count and limited diagnostics are noted.
Use the result arrays/JSON to inspect all values. Panels are bounded summaries,
not exhaustive logs; `to_dict()` is the machine-readable scalar result.

Corner retains its existing specialized `render_debug(color, color_order=...,
show_contours=True, max_labels=12, panel=True)` and PNG save API. It also accepts `depth=...` with explicit scale when color is absent, using the
same percentile display as other results. Its specialized candidate-label options
remain compatible. Other results can also use recorded masks without RGB.

If diagnostics were disabled, known masks/results still render, but absent paths
or measurements are never invented. With no recorded spatial arrays, supply
color or depth for a limited summary. Existing Corner diagnostics-disabled results
show known candidate points and a limited-view notice.

Saving supports PNG, an existing non-symlink parent directory, and no overwrite
by default. Existing destination files/symlinks are refused by default. `overwrite=True` uses
atomic directory-entry replacement; it never follows a destination symlink into
another file. The link is replaced, not its target.
Errors are explicit. Paths are not embedded as image metadata. Large panels can
contain caller-supplied labels such as pose keys; use non-sensitive identifiers.

Adjustment order: validate scale/registration/masks and reference pose first;
inspect missing/uncertain coverage; adjust spatial support to resolution; then
metric/color thresholds using positives **and negatives**. Inspect each feature
independently. Fewer unknowns is not necessarily higher correctness. See API,
GRASP, OBJECT_GRASP, REFERENCE, SURFACE and CORNER for each knob and limits.
Run `python examples/example_debug_results.py --output-dir EXISTING_DIRECTORY`.

## Overhead internal boundaries

`OverheadBoundaryDetector` independently measures color foreground, local darkness,
depth steps and relative-depth islands. See [BOUNDARY.md](BOUNDARY.md) for inputs,
parameter units, stage uncertainty and public `render_debug` / `save_debug` methods
on `BoundaryResult`. `examples/example_boundary.py` is a minimal in-memory example.
Color evidence and relative relief do not establish wrinkles, layers or graspability.

OuterContour has `render_debug(color=None, *, color_order="RGB", depth=None,
depth_scale_m=.001, show_raw=True, show_ids=True, panel=True)`. Its inherited
`save_debug(path, color=None, *, color_order="RGB", overwrite=False, **options)`
forwards those render options and returns a Path. It does not accept the common
mask_alpha/max_lines controls. No optional diagnostics are required for known contours.
