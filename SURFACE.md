# Elevated and nonflat surface evidence

`DepthSurfaceDetector` identifies observable regions that may merit inspection
for piling, folds or raised material. It does not count layers or prove overlap.
A suspended single sheet, wrinkle or air gap can produce identical evidence.
`layer_status` is always `unknown`, even for a strong anomaly candidate.

```python
from depth_image_lens import DepthSurfaceDetector
result = DepthSurfaceDetector().detect(
    depth_uint16, object_mask, depth_scale_m=.001,
    intrinsics=intrinsics, table=table_plane, diagnostics=True)
image = result.render_debug(depth=depth_uint16)
# Explicit output only; parent directory must exist:
result.save_debug("surface.png", depth=depth_uint16)
```

`object_mask` is a boolean HxW visible deformable-object selection. Supply
`exclude_mask` for grippers/clutter and `observed_mask` for additional depth
validity. Raw zero/NaN/Inf remains missing. All arrays use the depth grid; RGB
is optional for rendering only. Scale is metres/count for every numeric dtype.
Intrinsics assume rectified optical-axis Z, not radial range.

Reference choices (mutually exclusive):
- `table=TablePlane(...)` with `CameraIntrinsics`: signed normal height in the
  camera frame. No external robot/world calibration. The plane's positive side
  must face above the table; use `fit_table_plane` with a verified table ROI.
- `reference_depth`: aligned same-pixel baseline, same scale, optional boolean
  `reference_valid`. Measures **camera-Z separation**, not normal thickness.
- Neither: independent local surface shape still runs, height remains unknown.

With intrinsics, a 3D SVD local-plane residual is measured. Without intrinsics,
local affine image-Z residual is an approximation; perspective can matter.
Adjacent Z steps are measured after reprojecting the fitted 3D plane onto camera
rays when intrinsics are present. This handles perspective on strongly oblique
planes. Without intrinsics the affine-Z approximation remains; `step_geometry`
records which path was used. Ordinary external object-table
boundaries are excluded by `boundary_guard_px`. Each connected object component
is fitted separately within fixed image cells. Cell masks are inspection regions,
not precise physical fold outlines. Steps exactly between cells and features
smaller than support requirements can be missed. Results depend on grid phase
and resolution; this is not automatic semantic segmentation.

| Parameter | Default | Units / purpose |
| --- | ---: | --- |
| `cell_px` | 24 | Processing-pixel local support size; scale when resizing. |
| `min_points` | 20 | Valid local samples / connected elevated support. |
| `min_valid_fraction` | .7 | Local selected-pixel coverage needed for shape. |
| `elevation_m` | .025 | Height/reference separation threshold. |
| `reference_error_m` | .005 | Caller-assumed height/reference error allowance. |
| `residual_m` | .005 | p95 absolute local-plane residual threshold. |
| `step_m` | .015 | Trend-corrected neighboring Z discontinuity threshold. |
| `min_step_pairs` | 3 | Required discontinuity pairs within one local cell. |
| `boundary_guard_px` | 1 | Erosion for mixed external-boundary depth, 0..3. |

Tune valid support and spatial scale first, then residual/step thresholds using
flat/sloped controls and sensor noise, then reference elevation/error. Do not
calibrate layer truth from height alone. Defaults are examples, not sensor bounds.

`state`: `anomaly_candidate`, `no_observed_anomaly`, `unknown`.
`elevation_state` and `shape_state` are independent. A valid local positive
survives unrelated missing regions; negative/clear results require adequate
observation. `metrics` includes height quantiles, coverage and pixel counts;
`regions` includes local residuals, step counts, support and failure reasons.
`elevated_mask`, `nonflat_mask`, `unknown_mask` are always returned. Missing shape
support is not a negative observation; known height can coexist with unknown
shape. `parameters` records effective thresholds. No calibrated probability is
returned. Flat layers with no observable relief remain indistinguishable.

Run `python examples/example_surface.py` for flat, raised-single-sheet and step
controls. See [debug views](DEBUGGING.md). Real overlap accuracy is unverified;
visible corner annotations and provisional masks are not overlap ground truth.
