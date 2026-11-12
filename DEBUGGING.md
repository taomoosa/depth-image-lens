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
| `BoundaryResult.filter_outer_proximity` | `BoundaryProximityResult` | Original/selected paths, exclusion band and artificial-border uncertainty; [options](BOUNDARY_PROXIMITY.md). |
| `OuterContourDetector` | `OuterContourResult` | Ordered straight/curve intervals, corners, clipping and section IDs; [options](OUTLINE.md). |
| `OverheadCornerDetector` | `CornerResult` | Specialized contour, candidate ID/status/reason overlay; [API](CORNER.md). |
| `OverheadObjectCounter` | `ObjectCountResult` | Per-group display colors/IDs, complete legend, unresolved count, and independent evidence masks; [API](OBJECT_COUNT.md#see-which-group-each-pixel-belongs-to). |

`analyze_color_geometry` returns ordinary component results; render those
individually. `CameraIntrinsics`, `TablePlane` and numeric difference helpers have
no spatial result renderer. Render their downstream results. There is no renderer
on every trivial data container.

The common render/save signatures are:

```python
result.render_debug(color=None, *, color_order='RGB', depth=None,
                    depth_scale_m=.001, mask_alpha=.35, panel=True, max_lines=32)
result.save_debug(path, color=None, *, color_order='RGB', overwrite=False, **options)
```

`save_debug` forwards the corresponding render options and returns a
`pathlib.Path`, not an image array. `render_debug` returns uint8 HxWx3 in the
requested RGB/BGR order; panel/tile dimensions can exceed the source image size.
Corner, Boundary, Outline and BoundaryProximity have the specialized options below;
other feature-specific results document their renderers in their own guides.

`ObjectCountResult` prepends a colored group overlay and solid ID map, with a
complete legend and explicit `count`/`candidate_count`/`state`. Its optional
`group_labels=True` draws spatial ID badges; `False` hides only those badges.
`panel=False` and `max_lines` do not remove or truncate the group legend.
Independent uncertainty, rejection and exclusion masks remain below the group
view, including when `count=None`. Group IDs/colors are result-local, not tracked
physical identities. See [group debugging](OBJECT_COUNT.md#see-which-group-each-pixel-belongs-to)
and `examples/example_object_count_debug.py`.

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
Use `json.dumps(result.to_dict(), allow_nan=False)` for strict JSON. Inspect
raw arrays separately with NumPy: unavailable entries may be NaN and must not
be serialized directly as JSON numbers.

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

```python
boundary.render_debug(color=None, *, color_order='RGB', depth=None,
                      depth_scale_m=.001, topology_labels=False, **options)
boundary.save_debug(path, color=None, *, color_order='RGB', overwrite=False, **options)
```

`topology_labels` is display-only; other options use the common renderer.


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

BoundaryProximity uses the specialized OuterContour-style render signature, with
`show_ids=False` and no `show_raw` option. See its guide for exact options and labels.

## Bridge-drape diagnostics

Opt in per call; normal JSON output remains unchanged when diagnostics are off:

```python
result = detector.detect_auto(depth, difference, upper_band=(8, 20), diagnostics=True)
# Manual mode also accepts diagnostics=True.
summary = result.to_dict()
arrays = result.diagnostics
```

This option does not change classification. The existing foreground, candidate-region and invalid masks remain on the result. Diagnostic masks/points are allocated only when requested; no image dependency is required by the core.

| Diagnostic | Meaning |
|---|---|
| `candidate_search_mask` | Visible eligible region below the automatic upper band, before foreground filtering. |
| `search_mask` | Manual/inferred anchor corridor used by the final detector, including positions that may be invalid. Compare with `invalid_mask`. |
| `left_anchor_mask`, `right_anchor_mask` | Anchor bands used for component connection tests. |
| `bridge_corridor_mask` | Observed pixels in components reaching both anchors. This is a candidate component, not a unique shortest path or verified material correspondence. Invalid pixels and tolerated gaps are not filled. |
| `sag_path_xy`, `sag_chord_xy` | Observed upper-envelope sample points and anchor chord in image xy coordinates. Missing bins have no sample. |
| `sag_path_used` | Whether the sampled envelope actually produced a sag classification. Points shown when false are diagnostic observations only. |
| `left_drape_component_mask`, `right_drape_component_mask` | Components connected to each anchor. These masks can overlap completely for a single bridge; they are not a left/right segmentation. |
| `opposite_search_mask` | Where an opposite candidate could be searched after one observed neck. It is a search region, not an invented hand. |

Some entries are absent if detection stops before that stage. Reasons and `auto.stage` explain where it stopped. `result.to_dict()` summarizes bool masks by shape and pixel count; point arrays become lists. `include_masks=True` expands bool masks to lists as well. Direct NumPy arrays remain available in `result.diagnostics`.

### Save arrays, JSON and an overlay

After installing the package, from the repository root:

```bash
python examples/example_debug.py --output debug-output
python examples/example_debug.py --depth depth.npy --difference difference.npy --config examples/config.auto.json --output debug-output
```

The no-input example is a synthetic U-shaped deformable-object bridge. The output directory receives:

- `arrays.npz`: foreground/invalid/candidate masks and available diagnostic arrays. Load with `np.load(path, allow_pickle=False)`.
- `result.json`: classifications, reasons, features and compact diagnostic metadata.
- `overlay.svg`: four panels for foreground/invalid pixels, search/bridge/anchors, sag measurement, and anchor-connected drape components.

Orange/blue distinguish the left/right anchor components; green marks a bridge component; red marks invalid depth. Black boxes are unverified hand candidates. The sag panel plots individual observed points, so missing bins are not connected by a fabricated line. SVG generation uses only NumPy and the Python standard library. Large masks may create large SVGs; use a consistent crop before detection. Reusing an output directory replaces these three diagnostic files. Each is written to a fresh temporary file in the same directory and published with `os.replace`: an existing output-file symlink is replaced as a directory entry, and its target is not opened or changed. The output directory itself must not be a symlink. Symlinks in ancestor directories follow normal filesystem resolution; choose a trusted parent directory. This is per-file replacement, not an all-three-files transaction or protection against concurrent hostile parent-directory renames.

These are visible-shape observations, not proof of physical holding or tension. Preserve depth sign, units, registration and invalid-background provenance when inspecting results.

### Shared CLI output

`depth-image-lens` uses the same `depth_image_lens.diagnostics.save_debug(result, output)` function as `examples/example_debug.py`; there is one diagnostic writer. It accepts separate NPY inputs or a numeric NPZ pair through the CLI described in [README.md](README.md). Result JSON never expands masks by default and is serialized with `allow_nan=False`. Only diagnostic arrays are saved in the NPZ; original depth files are not copied. Generated files contain no source paths or arbitrary config metadata.

Continuity diagnostics: the search panel additionally displays `inferred_gap_mask` in yellow and `depth_rejected_mask` in pink. Yellow is explicitly inferred connectivity through known non-foreground pixels, not observed foreground or hole filling. JSON `auto.candidate_evaluations` explains the bounded candidate ranking and rejected depth paths. Sag dots continue to show observed envelope samples only.
