# Practical parameter and debug reference

This is the cross-class tuning index for the public `depth_image_lens` API.
Defaults below are implementation defaults, **not verified sensor/site settings**.
No setting establishes physical holding, safe grasping, tension or hidden layer count.
Change one evidence source at a time, preserve a baseline, and compare positive,
negative and missing-data cases. Fewer unknowns alone is not improvement.

## RGB-D background masks and overhead pick candidates

See [the complete RGB-D/pick tuning guide](RGBD_PICK_TUNING.md) for every
constructor setting, default, type, range, unit, adjustment direction and
interaction, plus input-mask contracts, symptoms and debug interpretation.
Run `python examples/example_rgbd_pick_tuning.py --output-dir debug-output/run-01`
with an existing `debug-output` parent. The example also accepts registered
current/empty NPZ inputs, JSON parameter overrides, a contour ID and a pixel to
inspect; it saves masks, representative points and numeric diagnostics.

## Installation and common input contracts

From the repository root in an isolated Python environment:

```bash
python -m pip install -e '.[debug]'
python examples/example_background_tolerances.py
mkdir -p debug-output
python examples/example_debug_results.py --output-dir debug-output
```

The debug, boundary and corner extras provide the same optional OpenCV dependency;
other numerical detectors use NumPy. If dependencies already exist, the offline
editable check is `python -m pip install --no-index --no-build-isolation --no-deps -e .`.
It does not test online dependency resolution. Use a fresh output directory on a
second debug run: saving intentionally refuses existing files.

- Color is NumPy `uint8`, HxWx3, explicit `color_order="RGB"` or `"BGR"`.
  Depth is HxW **optical-axis Z**, not radial range. Color/depth registration and
  synchronization are caller responsibilities; matching array sizes are insufficient.
- `detect_16uc1` convenience methods require HxW `uint16`; `depth_scale_m=.001`
  means 1 count = 1 mm. General scale-based methods use the declared scale for
  **every dtype** (use 1.0 for metres). Unit-based methods use `depth_unit="m"`
  or `"mm"`; Lift's general `detect` specifically receives `depth_m` in metres.
- Raw depth 0, NaN, Inf and `observed_mask=False` are missing, not empty space.
  Positive clipped difference is `max(empty-current,0)`, computed after casting
  to signed/float arithmetic. Difference zero is valid non-foreground/exclusion,
  except where a separate validity mask says unavailable. Do not reconstruct a
  physical empty depth from a masked positive-only difference.
- Region/seed/validity masks are bool HxW, in the same cropped grid. Convert a
  binary uint8 mask with `mask != 0`. An object mask excludes hands/table; the
  Bridge foreground mask may contain candidate supports and needs arm exclusions.
  Hand dictionaries use `left` / `right` image-side labels, not robot identity.
- Arrays index `[y,x]`; points use `(x,y)`; boxes `(x0,y0,x1,y1)` exclude the upper
  bounds. Pixel lengths must follow image resizing; pixel areas scale quadratically.
  Metric thresholds do not scale with resolution. Update intrinsics after crop/resize.
- `CameraIntrinsics(fx,fy,cx,cy)` uses pixel focal lengths and principal point;
  `intrinsics.cropped(x0,y0)` handles an unscaled crop. Distortion must already be corrected.
  `TablePlane(normal, offset_m)` normalizes coefficients and computes signed camera-frame
  distance. Above-table is positive. Neither class estimates robot/world calibration.

## Choose evidence before thresholds

| API | Required evidence / useful optional inputs | Meaning and limitation |
|---|---|---|
| BridgeDrapeDetector | Depth+difference, or depth+foreground mask; manual hand boxes or auto upper search band; optional arm/work masks | Image bridge/sag/drape and support hypotheses; not physical contact. Partial independent measurements can survive conservative unknown decisions. |
| IndependentGraspDetector | Depth, object mask, two hand masks, registration declaration; optional observability/uncertainty | Per-side likely/unlikely/unknown from local matched contact patches. Does not consume bridge results or prove holding. |
| ObjectOnlyGraspDetector | Depth+object mask and either difference reference or intrinsics+table; optional gripper/exclusion masks | Rule-based held/bilateral inference, not observed grasp. No meaningful reference gives unknown. Width failure is not single-hand proof. |
| LiftDetector / fit_table_plane | Object/empty-table mask, depth and intrinsics; Lift additionally needs oriented table plane | Visible elevation versus whole-object lift. Plane fit needs caller-selected actual table; low residual cannot recognize a wall. |
| OverheadCornerDetector | Color and depth argument, supplied object mask or optional empty color; explicit mode/registration | Outer convex angle, line support and local depth checks. No table-plane height or physical usable grasp-area measurement. |
| SeededColorAssistant | Aligned color, foreground and object/hand color seeds; optional validity | Appearance candidates and unknowns, not learned semantic hand segmentation. |
| ReferenceGripperSegmenter | Independent empty/gripper-only depth, coarse hand boxes, static ROI and pose key; optional paired color | Fixed/repeatable-pose reference matching. No automatic pose registration/bank selection; changed pose requires matching reference. |
| DepthSurfaceDetector | Depth+object mask; optional intrinsics and table or reference depth | Local nonflat/elevation hypotheses; no reference still permits shape evidence. Layer count stays unknown. |
| OverheadBoundaryDetector | Color; optional depth, explicit object mask, table samples, rectangle or independent empty color | Color bands, depth steps/islands, optional graph. Prints/shadows may mimic folds. Automatic empty domain can produce explicitly unscoped depth evidence. |

Existing observed-table fits are references, not ground-truth physical calibration.
A single wide lifted sheet can produce bilateral inference while only one hand holds it.
IndependentGrasp local evidence and ObjectOnly inferred evidence are intentionally separate.

## Parameter tables

Each table lists every constructor setting (or plane-fit keyword). `float` means a
finite real-valued threshold, `int` means processing-pixel/count setting, and `bool`
is an explicit switch. Selection modes are strings. Increasing/decreasing effects
are conditional on earlier validity/reference checks; they are not promises of
monotonic end-to-end correctness. Constructor validation is not relaxed by missing
optional image evidence. See linked contracts for permitted ranges and failure reasons.

### BridgeDrapeDetector

[Detailed contract](API.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `threshold_m` | `0.02` | float; m | Positive foreground must exceed this depth difference. Increase to reject smaller changes; decrease admits noise. Unused by detect_mask. |
| `min_area` | `20` | int; pixels (area) | Minimum foreground component support. Increase rejects small regions, including thin targets. |
| `anchor_band_px` | `3` | int; pixels | Thickness retained at candidate contact anchors. Increase admits more attachment support and possible arm clutter. |
| `max_gap_px` | `0` | int; pixels | Maximum small observed gap for component connectivity; zero disables it. Increase permits inferred links only within validity/depth guards, not missing-depth fill. |
| `sag_ratio` | `0.15` | float; dimensionless | Large-sag threshold relative to anchor span, measured from the upper envelope. Increase requires deeper apparent sag; hem length is not sag. |
| `drape_ratio` | `0.25` | float; dimensionless | Minimum downward extent relative to anchor span. Increase requires longer hanging evidence. |
| `static_hanging` | `True` | bool; bool assumption | True declares static hanging; False withholds hanging interpretations. This is a scene assumption, not a tuning shortcut. |

### IndependentGraspDetector

[Detailed contract](GRASP.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `contact_radius_px` | `3` | int; pixels | Integer 1..8 nearest-boundary search radius. Increase admits more separated candidate contacts; not physical contact proof. |
| `depth_tolerance_m` | `0.04` | float; m (camera Z) | Maximum local hand/object depth difference. Increase admits larger discontinuities and accidental proximity. |
| `min_contact_pixels` | `3` | int; pixels per connected side | Minimum support on both sides of one corresponding patch. Increase needs more support; disconnected small contacts are never pooled. |
| `min_consistent_fraction` | `0.6` | float; fraction | Minimum depth-consistent fraction within a corresponding contact patch. Increase rejects mixed-depth patches, not unrelated patches. |
| `min_valid_fraction` | `0.8` | float; fraction | Coverage needed for a reliable negative. Increase makes negative claims harder; a separately valid local positive can survive unrelated missing data. |

### ObjectOnlyGraspDetector

[Detailed contract](OBJECT_GRASP.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `min_lift_m` | `0.03` | float; m | Required separation after reference_error_m allowance. Increase requires stronger elevation evidence; background difference is camera-Z separation, not table-normal height. |
| `min_width_px` | `40` | int; pixels | Minimum contiguous lifted row span for bilateral inference. Increase needs wider support within one component; no summing disconnected widths. |
| `min_pixels` | `20` | int; pixels (area) | Minimum valid lifted component support. Increase removes small candidates/noise. |
| `min_width_rows` | `3` | int; consecutive rows | Number of rows with sufficient span. Increase rejects isolated wide rows; image orientation matters. |
| `min_lift_fraction` | `0.1` | float; fraction | Minimum lifted component support relative to valid points in its observation region. Increase requires a larger local lifted share. |
| `min_valid_fraction` | `0.7` | float; fraction | Minimum valid coverage within the relevant connected observation region. Increase withholds inference for missing regions; unrelated regions do not dilute a valid one. |
| `reference_error_m` | `0.005` | float; m | Caller-declared reference uncertainty allowance. Increase reduces confident elevation and near-reference claims; not an estimated sensor guarantee. |
| `depth_jump_m` | `0.1` | float; m (camera Z) | Maximum adjacent jump for a lifted component. Increase can merge unrelated surfaces; lowering can fragment real relief. |

### LiftDetector

[Detailed contract](API.md#table-relative-lift).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `lift_height_m` | `0.03` | float; m (table normal) | Required height after height_error_m allowance. Increase requires stronger lift evidence. |
| `near_table_m` | `0.01` | float; m (table normal) | Near-table band. Increase admits more elevated points as near-table, affecting negative/whole-object claims. |
| `height_error_m` | `0.005` | float; m | Declared height error bound. Increase makes confident elevation/near-table claims harder; it is not fitted uncertainty. |
| `min_pixels` | `20` | int; pixels (count) | Minimum selected valid support. Increase withholds decisions on small observations. |
| `min_component_pixels` | `12` | int; pixels (area) | Minimum connected component support for component evidence. Increase suppresses small near-table/elevated regions. |
| `min_lift_fraction` | `0.1` | float; fraction | Required elevated share of valid selected pixels. Increase requires broader observed elevation. |
| `min_valid_fraction` | `0.7` | float; fraction | Minimum valid observation coverage. Increase requires more complete depth before the corresponding aggregate decision. |

### OverheadCornerDetector

[Detailed contract](CORNER.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `min_area_px` | `100` | int; pixels (contour area) | Minimum external component contour area. Increase removes small components, not just corners. |
| `polygon_epsilon_px` | `2.0` | float; pixels | Base approxPolyDP approximation tolerance (not perimeter fraction). Larger smooths contour details; smaller can oversegment noise. |
| `edge_window_px` | `18` | int; pixels | Maximum local edge-fit support length. Larger uses more context but can include curvature or neighboring corners. |
| `angle_tolerance_deg` | `25.0` | float; degrees around 90 | Legacy shared angle tolerance used when a separate proposal/fitted setting is None. Increase admits less right-angled geometry. |
| `line_residual_px` | `1.0` | float; pixels (RMS) | Maximum local fitted-line residual. Increase accepts less straight/noisier edges. |
| `depth_radius_px` | `24` | int; pixels | Corner depth neighborhood radius. Larger increases support but may include folds/other local geometry; component scope is preserved. |
| `depth_step_m` | `0.015` | float; m | Internal depth-step threshold. Increase tolerates larger internal steps, potentially admitting overlap evidence. |
| `depth_residual_m` | `0.005` | float; m | Maximum local plane residual. Increase accepts less planar geometry. Not table proximity or layer truth. |
| `color_distance` | `35.0` | float; 8-bit RGB distance | Foreground change threshold when background_color supplies Corner segmentation. Increase requires stronger RGB change; differs from Boundary's encoded Lab units. |
| `support_policy` | `'fixed'` | str; string | fixed or adaptive. Adaptive uses available local edge extent for support; not a continuous confidence control. |
| `proposal_policy` | `'single'` | str; string | single or multiscale. Multiscale tries epsilon factors 0.5,1,2,4; raises candidate coverage and potential unmatched detections. |
| `proposal_merge_px` | `3.0` | float; pixels | Same-component proposal/refined-candidate deduplication radius. Increase can merge nearby genuine corners; decrease retains duplicates. |
| `proposal_angle_tolerance_deg` | `None` | float or None; degrees or None | Initial polygon angle tolerance around 90. None inherits angle_tolerance_deg. Increase broadens proposals without relaxing fitted angle. |
| `fitted_angle_tolerance_deg` | `None` | float or None; degrees or None | Refined angle tolerance around 90. None inherits angle_tolerance_deg. Increase accepts non-right-angle line geometry. |
| `angle_stability_mode` | `'reject'` | str; string | reject enforces long/short-window angle agreement; diagnostic records it without rejecting. Other guards remain active. |
| `angle_stability_deg` | `10.0` | float; degrees | Maximum long/short-window angle difference. Increase loosens only rejection mode; diagnostic mode records the comparison. |
| `max_intersection_shift_px` | `2.0` | float; pixels | Maximum long/short-window fitted intersection displacement. Increase allows unstable/localization-shifting fits. |
| `min_fit_support` | `4` | int; sample count | Minimum points on each side in both fit windows (at least 3). Increase improves support requirements but loses short/cropped edges. |
| `boundary_mixing_m` | `None` | float or None; m or None | Outer boundary mixed-depth threshold. None inherits depth_step_m; set independently to avoid coupling internal steps and boundary mixing. |

### SeededColorAssistant

[Detailed contract](RGBD.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `max_color_distance` | `60.0` | float; 8-bit RGB distance | Maximum nearest seed-palette distance. Increase covers more appearance variation and possible wrong materials. |
| `min_color_margin` | `15.0` | float; 8-bit RGB distance | Required advantage over the other seed class. Increase returns more appearance uncertainty. |
| `boundary_guard_px` | `1` | int; pixels | Integer 0..3 erosion of assigned interiors. Increase withholds more boundary pixels; does not fix misregistration. |

### ReferenceGripperSegmenter

[Detailed contract](REFERENCE.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `foreground_m` | `0.02` | float; m (camera Z) | Empty-minus-gripper/current threshold. Increase requires stronger foreground changes and can miss thin geometry. |
| `match_depth_m` | `0.01` | float; m (camera Z) | Maximum current/reference gripper depth mismatch. Increase admits pose or object contamination. |
| `match_color_distance` | `45.0` | float; 8-bit RGB distance | Maximum optional current/reference gripper color distance. Increase tolerates illumination and less distinctive colors. |
| `static_depth_m` | `0.01` | float; m (camera Z) | Static-region depth check tolerance. Increase allows greater scene/camera drift; does not align images. |
| `static_color_distance` | `35.0` | float; 8-bit RGB distance | Static-region color check tolerance. Increase allows appearance drift, not proof of correct reference. |
| `min_pixels` | `20` | int; pixels (count) | Minimum reference/matched support. Increase requires larger visible reference regions. |
| `min_valid_fraction` | `0.7` | float; fraction | Minimum available reference/current coverage. Increase abstains more often on missing/offscreen regions. |
| `min_match_fraction` | `0.7` | float; fraction | Minimum expected reference-hand match. Increase requires stronger agreement; sides are checked independently. |
| `max_unexpected_fraction` | `0.05` | float; fraction of reference hand area | Maximum unexpected occupancy in a search region. Increase tolerates more clutter/pose mismatch, risking contaminated masks. |

### DepthSurfaceDetector

[Detailed contract](SURFACE.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `cell_px` | `24` | int; pixels | Fixed inspection-cell side length. Larger fits broader regions and can smooth small relief; grid placement affects evidence. |
| `min_points` | `20` | int; point count | Minimum valid points per component/cell fit. Increase makes sparse regions unknown. |
| `min_valid_fraction` | `0.7` | float; fraction | Minimum valid coverage within local selected cell/component. Increase requires fuller local observation. |
| `elevation_m` | `0.025` | float; m | Required height/separation above supplied reference after error allowance. Increase needs stronger elevation evidence. |
| `reference_error_m` | `0.005` | float; m | Declared reference error allowance. Increase makes elevation harder to confirm. |
| `residual_m` | `0.005` | float; m | Local-plane p95 residual threshold for nonflat evidence. Increase tolerates more local relief/noise. |
| `step_m` | `0.015` | float; m (camera Z) | Detrended adjacent-step threshold. Increase suppresses weaker nonflat evidence; with intrinsics, plane-ray detrending accounts for perspective. |
| `min_step_pairs` | `3` | int; adjacent pair count | Minimum above-threshold pairs for step evidence. Increase rejects isolated noise/short steps. |
| `boundary_guard_px` | `1` | int; pixels | Excluded outer component band (0..3). Increase avoids mixed outer edges but loses narrow regions. |

### OverheadBoundaryDetector

[Detailed contract](BOUNDARY.md).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `color_distance` | `30.0` | float; encoded 8-bit Lab distance | Legacy/table-palette color separation threshold. Increase requires stronger foreground differences; unused for separate_tolerances selection but still used by fallback. |
| `color_margin` | `8.0` | float; encoded 8-bit Lab distance | Legacy/table-palette uncertainty band around color_distance. Increase marks more uncertain pixels; does not set the separate score margin. |
| `min_object_area` | `100` | int; pixels (area) | Minimum selected connected component area. Increase removes small objects; explicit filtered masks never expand to the frame. |
| `border_px` | `5` | int; pixels | Table sampling border when no better domain/reference is supplied. Increase samples more border; a target on the border violates the assumption. |
| `boundary_guard_px` | `3` | int; pixels | Excluded component-exterior band for internal evidence. Increase suppresses outer-edge artifacts and can lose near-edge folds. |
| `color_window_px` | `9` | int; pixels | Odd 3..101 grayscale closing window for darkness. Larger detects broader dark bands but changes local background and may include shading. |
| `darkness_threshold` | `12.0` | float; intensity 0..255 | Closing-minus-gray threshold. Increase removes weaker dark bands; selected by OR with gradient. |
| `gradient_threshold` | `12.0` | float; intensity per pixel | Sobel magnitude/8 threshold. Increase removes weaker transitions; bright edges, prints and shadows also qualify. |
| `min_line_pixels` | `6` | int; pixels (band area) | Minimum connected color evidence-band support. Increase discards short/thin fragments, not a physical line-length threshold. |
| `depth_step_m` | `0.01` | float; m (camera Z) | Detrended adjacent step threshold; effective lower bound is twice depth_noise_m. Increase retains only larger steps. |
| `depth_noise_m` | `0.002` | float; m | Declared noise floor. Increase suppresses weak depth steps/islands even if their nominal thresholds are smaller. |
| `island_height_m` | `0.01` | float; m | Minimum relative residual/contrast for raised or depressed patches, subject to noise floor. Increase requires stronger relief; not table height. |
| `min_island_area` | `25` | int; pixels (area) | Minimum connected island support. Increase rejects small patches/noise. |
| `max_island_fraction` | `0.35` | float; fraction of component | Maximum island share. Increase admits broader relief but weakens the dominant-baseline interpretation. |
| `association_px` | `3` | int; Chebyshev pixels | Color-band proximity to depth steps. Increase raises agreement without establishing common physical identity. |
| `island_context_px` | `5` | int; pixels | Surrounding annulus for local baseline/contrast checks. Larger uses wider context, possibly other relief. |
| `grabcut_iterations` | `3` | int; iterations | Integer 1..10 for optional rectangle segmentation. More iterations cost time and need not improve correctness; labels remain provisional. |
| `background_compensate` | `False` | bool; bool | Enable one additive median encoded-Lab offset from background_check_mask. Requires valid unchanged table support; not gain correction/local shadow removal. |
| `background_check_p95` | `20.0` | float; encoded Lab distance | Maximum stable-table residual p95. Increase accepts poorer references; Tc/Tl do not relax this check. |
| `background_max_offset` | `40.0` | float; encoded Lab distance | Maximum compensation vector norm. Increase permits larger global corrections, risking mismatched reference acceptance. |
| `background_difference_mode` | `'lab_distance'` | str; string | lab_distance preserves legacy selection; separate_tolerances enables the seedless elliptical Lab test. |
| `background_chroma_tolerance` | `30.0` | float; encoded Lab chromatic distance | Finite positive Tc. Larger tolerates more chromatic difference and can hide colored objects. |
| `background_luma_tolerance` | `30.0` | float; encoded Lab luminance distance | Finite positive Tl. Larger tolerates brightness/shadows and can completely hide brightness-only objects. |
| `background_score_margin` | `0.26666666666666666` | float; dimensionless radius | 0..1 uncertainty band abs(sqrt(S)-1) <= margin. Increase marks more uncertainty, not a change in foreground S>1. |
| `background_shadow_mode` | `'off'` | str; string | off or uniform_color. The latter additionally needs object_color_seed and valid distinguishable uniform color support; seedless calls do not gain suppression. |
| `shadow_chroma_threshold` | `0.04` | float; normalized encoded-RGB chroma distance | Tolerance for table-like shadow appearance and seed/reference uniformity checks. Increase both broadens similarity and demands greater seed separation; not monotonic recall. |
| `shadow_intensity_floor` | `20.0` | float; 8-bit RGB intensity | Minimum useful intensity for chroma comparison. Increase withholds dark unreliable samples, potentially reducing suppression. |
| `shadow_min_ratio` | `0.35` | float; dimensionless intensity ratio | Lower allowed current/reference intensity ratio. Lower admits deeper shadow hypotheses, risking dark-object ambiguity. |
| `shadow_max_ratio` | `1.15` | float; dimensionless intensity ratio | Upper allowed current/reference intensity ratio. Higher permits more brightening hypotheses. Suppressed candidates remain unknown, not confirmed table. |
| `trace_boundaries` | `False` | bool; bool | Enable optional ordered centerlines/graph; does not replace original measured masks. |
| `color_weak_ratio` | `1.0` | float; dimensionless (0,1] | Multiply strong darkness/gradient thresholds for seeded weak continuation. Lower connects more weak evidence and texture; isolated weak components are discarded. |
| `depth_weak_ratio` | `1.0` | float; dimensionless (0,1] | Multiply step threshold for weak continuation, retaining twice-noise floor. Lower retains weaker connected steps, not missing depth. |
| `max_gap_px` | `0` | int; pixels (integer 0..20) | Maximum missing centerline samples; 0 disables interpolation. Larger permits more short hypotheses subject to unique partner/tangent/domain/crossing guards. |
| `gap_angle_deg` | `25.0` | float; degrees (0..60) | Maximum both-end direction error relative to connector. Increase allows less collinear reconnection; ambiguous partners remain separate. |
| `corner_turn_deg` | `45.0` | float; degrees (0,180] | Turn threshold on a degree-two path. Lower splits gentler bends; values above 90 may retain right angles within a curve. |
| `corner_window_px` | `5` | int; pixels arclength (2..50) | Chord support and corner suppression scale. Larger measures broader bends; interacts with turn threshold and nearby corners. |
| `tangent_window_px` | `5` | int; pixels arclength (2..50) | Directional support for reconnection/junction arms. Larger needs longer observed paths and changes angle estimates. |
| `min_path_length_px` | `3.0` | float; pixels arclength | Shorter graph segments are marked short_path, not deleted. Increase changes short-path status, not original evidence. |
| `junction_cluster_px` | `0` | int; pixels arclength (integer 0..20) | Optional degree-three pair grouping; 0 off. Larger considers longer observed connectors and moves fit support outward. Raw nodes remain unchanged. |
| `junction_cluster_residual_px` | `1.0` | float; pixels (0,3] | Maximum grouping arm/intersection residual. Increase risks grouping distinct nearby T nodes. |
| `junction_cluster_angle_deg` | `15.0` | float; degrees (0..45) | Maximum deviation of opposing arms from 180. Increase loosens grouping hypotheses, not confirmed physical junctions. |

### fit_table_plane

[Detailed contract](API.md#empty-table-plane-helper).

| Parameter | Default | Type / unit | Meaning and direction |
|---|---|---|---|
| `depth_scale_m` | `0.001` | float; m/count | Finite positive conversion for every dtype; .001 for mm, 1 for m. This is calibration, not an accuracy knob. |
| `positive_direction` | `None` | vector or None; 3-vector or None | Camera-frame positive normal direction. None faces camera origin. Set actual above-table side; do not flip to obtain a preferred answer. |
| `min_points` | `20` | int; point count | At least 3 valid table points; increase requires more support, not automatic contamination protection. |
| `min_spread_m` | `0.01` | float; m (in-plane RMS) | Minimum second in-plane spread. Increase rejects narrow/near-collinear ROIs. |
| `residual_warning_m` | `0.005` | float; m | Residual warning threshold only; decrease produces more warnings, not fit rejection. |
| `diagnostics` | `False` | bool; bool | Retain table ROI/support for debug. Does not change the fitted plane. |

### Corner starting preset and coupled controls

`OverheadCornerDetector.recommended_for_tuning()` selects `proposal_policy="multiscale"`,
`support_policy="adaptive"`, `angle_stability_mode="diagnostic"`. It retains other
constructor defaults, including `min_fit_support=4`. A prior external numerical
selection of support6 is **not** the constructor/preset default or a real-robot optimum.
Pass explicit overrides, e.g. `recommended_for_tuning(min_fit_support=6)`, only after
checking short/cropped true corners and unmatched detections in your setup.
Separate proposal/fitted tolerances and `boundary_mixing_m` from their legacy fallback
controls before independent tuning. Diagnostic stability removes only that rejection;
convexity, support, geometry, visibility and requested depth gates remain.

### Boundary: difference, shadow and topology tuning order

1. Validate the object domain and reference alignment before inspecting internal lines.
   Explicit `object_mask` wins; empty masks remain empty. Automatic border-table sampling
   is unsuitable when the object occupies the border. Manual table samples/rectangle can
   help; none guarantees semantic segmentation. Reference checks need >=25 unchanged
   table pixels and cannot certify that the rest of the reference is empty.
2. For background difference, default `lab_distance` uses `color_distance`.
   Opt-in `separate_tolerances` uses `S=(dC/Tc)**2+(dL/Tl)**2 > 1` after optional
   additive global compensation. Equal tolerances reproduce the legacy selection.
   Raise Tl modestly for brightness changes while testing dark/brightness-only objects;
   raise Tc only for measured chromatic variation. These are OpenCV encoded Lab units,
   not calibrated Delta E. Inspect dC/dL/score separately. Margin is in sqrt(S), not S.
3. Reference checks remain unweighted Lab; raising Tl does not bypass them. Rejected or
   malformed references use an uncertain fallback; a wholly invalid shaped reference
   provides no foreground. Without checks, global change can select the entire frame.
4. Optional `uniform_color` shadow mode comes after foreground selection and requires
   a small distinguishable object-color seed. It cannot restore objects lost by large
   Tc/Tl. Suppressed shadow hypotheses remain **unknown**, not confirmed table. Distinct
   material colors need not remain distinguishable under shadow/exposure/clipping.
5. Tune closing/guard windows for resolution, then strong darkness/gradient OR thresholds.
   Tune metric step/noise and island context separately. Agreement is spatial proximity,
   not a calibrated probability. Intrinsics change island residual to normal distance;
   depth-step thresholds always use detrended camera-Z differences.
6. Enable graph tracing with weak ratios1/gap0 first. Lower ratios for connected weak
   evidence; enable short gaps last. Original observed masks stay unchanged. Orange
   inferred connections have no invented darkness/depth measurement. Lower corner-turn
   threshold splits gentler curves; junctions are connectivity nodes, not angle corners.
   Junction grouping defaults off and only adds hypotheses, preserving raw topology.

[Difference details](BACKGROUND_TOLERANCES.md), [shadow assumptions](BOUNDARY_SHADOW.md),
[topology options and ranges](BOUNDARY_TOPOLOGY.md). Real paired empty/current accuracy
has not been established by reconstructed references; mask-assisted proxy evaluation
contains target-information leakage. Use independent site captures and labels.

## Call-time controls and independent continuation

Constructor parameters alone do not describe the evidence. The exact signatures below
list public call-time defaults. Required arguments have no default. `**options` passes
only documented controls to the corresponding general method, not arbitrary settings.

- `diagnostics=False`: optional intermediate arrays are not retained. Set True while
  tuning; it does not authorize filling missing evidence or changing thresholds.
- `observed_mask`, `valid_mask`, `color_valid_mask`, `background_valid` and
  `reference_valid`: bool availability in the declared grid; False means unknown,
  not observed background. Color validity is distinct from depth validity.
- `hand_visibility`: each side `visible`, `partial` or `hidden` (including fully offscreen). Defaults are declarations, not automatic validation. `mask_complete`
  and `whole_object_visible` must reflect real observation, not a desired decision.
  Negative claims need sufficient valid completeness; local positive evidence can be
  returned with global partial-view warnings. Inspect per-side/per-component results.
- Bridge `down=(0,1)` is normalized image-down, not world gravity. `upper_band=None`
  uses the top quarter projected along down. `min_span_px=30` sets minimum candidate
  separation; larger rejects narrower spans. `max_height_delta_px=12` is allowed
  anchor height mismatch; larger admits less same-height geometry. `depth_jump_m=.1`
  limits connected depth jumps. `anchor_policy="local_observed"` requires observed
  3x3 support and max_gap_px0; it does not invent cropped contact.
- Bridge `difference_mode="signed"` and sign describe actual subtraction.
  `positive_foreground` is the pre-masked/clipped contract; the uint16 wrapper chooses it.
  Supplied-mask mode bypasses difference thresholding and uses `mask_complete=False`
  by default. Do not reclassify excluded pixels as missing depth to force connectivity.
- `hand_rois` / `hand_regions` / `hand_masks` are distinct boxes, assigning regions and
  semantic candidate masks. `detect_color_candidates` needs hand regions to assign
  the appearance hand class; it does not localize two hands automatically.
- ObjectOnly `gripper_masks` and `exclude_mask` remove non-object pixels before area/span
  measurement; `uncertain_mask` withholds measurement without pretending complete
  coverage. Area/width ratios to visible grippers are diagnostic-only; partial grippers
  make these lower-bound image extents, not physical hand-size calibration.
- Surface: choose `table` with intrinsics OR `reference_depth`, not both. Reference depth
  measures camera-Z separation, not thickness. Without either, elevation is unknown
  while local shape can continue. ObjectOnly likewise needs a meaningful reference for
  held inference; absolute camera depth or object pixel height alone is insufficient.
- Boundary: `background_valid_mask` is in reference coordinates **before** the known
  integer `background_shift_xy`; `background_check_mask` is in current coordinates.
  Positive dx/dy moves reference right/down. No transform is estimated. `table_mask`
  samples visible table; `object_color_seed` samples current object interior;
  `object_roi` is a coarse rectangle, `depth_roi` an independent bool depth domain.
  Optional unusable depth/reference is reported stagewise, leaving usable color stages.
  Fatal shared color/explicit-mask shape errors still raise. Unscoped fallback depth is
  explicitly marked and is not internal-object evidence.
- Reference segmenter `pose_key` must match the calibrated acquisition; it is not a
  tracker. Repeatable poses can use caller-selected separate reference objects. Only
  Boundary's known color translation is implemented; this does not align gripper poses.

### BridgeDrapeDetector call signatures

```python
BridgeDrapeDetector.detect(depth, difference, hand_rois, *, depth_unit='m', difference_unit='m', difference_sign='background-current', difference_mode='signed', background_valid=None, observed_mask=None, arm_mask=None, work_roi=None, hand_visibility=('visible', 'visible'), down=(0, 1), diagnostics=False, depth_jump_m=0.1, anchor_policy='complete')
BridgeDrapeDetector.detect_16uc1(depth, difference, hand_rois=None, *, depth_scale_m=0.001, **options)
BridgeDrapeDetector.detect_auto(depth, difference, *, upper_band=None, min_span_px=30, max_height_delta_px=12, depth_jump_m=0.1, down=(0, 1), diagnostics=False, **options)
BridgeDrapeDetector.detect_mask(depth, foreground_mask, hand_rois=None, *, depth_unit, observed_mask=None, arm_mask=None, work_roi=None, hand_visibility=('visible', 'visible'), down=(0, 1), diagnostics=False, depth_jump_m=0.1, anchor_policy='complete', mask_complete=False)
```

### IndependentGraspDetector call signatures

```python
IndependentGraspDetector.detect(depth, object_mask, hand_masks, *, depth_unit, registered_to_depth, observed_mask=None, uncertain_mask=None, hand_visibility=('visible', 'visible'), mask_complete=(False, False), diagnostics=False)
IndependentGraspDetector.detect_16uc1(depth, object_mask, hand_masks, *, depth_scale_m=0.001, **options)
IndependentGraspDetector.detect_color_candidates(depth, colors, hand_regions, *, depth_unit, **options)
```

### ObjectOnlyGraspDetector call signatures

```python
ObjectOnlyGraspDetector.detect(depth, object_mask, *, depth_unit, difference=None, intrinsics=None, table=None, observed_mask=None, background_valid=None, whole_object_visible=False, mask_complete=False, diagnostics=False, gripper_masks=None, exclude_mask=None, uncertain_mask=None)
ObjectOnlyGraspDetector.detect_16uc1(depth, object_mask, *, difference=None, depth_scale_m=0.001, **options)
```

### LiftDetector call signatures

```python
LiftDetector.detect(depth_m, intrinsics, table, object_mask, *, exclude_mask=None, observed_mask=None, whole_object_visible=False, diagnostics=False)
LiftDetector.detect_16uc1(depth, intrinsics, table, object_mask, *, depth_scale_m=0.001, **options)
```

### OverheadCornerDetector call signatures

```python
OverheadCornerDetector.detect(color, depth, *, color_order, registered_to_depth, object_mask=None, background_color=None, depth_scale_m=0.001, valid_mask=None, intrinsics=None, mode='depth', diagnostics=False)
```

### SeededColorAssistant call signatures

```python
SeededColorAssistant.segment(color, foreground_mask, *, object_seeds, hand_seeds, color_order, registered_to_depth, valid_mask=None, color_valid_mask=None)
SeededColorAssistant.segment_16uc1(depth, difference, color, *, object_seeds, hand_seeds, color_order, registered_to_depth, depth_scale_m=0.001, threshold_m=0.02, observed_mask=None, background_valid=None, color_valid_mask=None)
```

### ReferenceGripperSegmenter call signatures

```python
ReferenceGripperSegmenter.calibrate(empty_depth, gripper_depth, hand_rois, static_mask, *, pose_key, depth_scale_m=0.001, gripper_color=None, color_order='RGB', registered_to_depth)
ReferenceGripperSegmenter.segment(depth, reference, *, pose_key, depth_scale_m=0.001, color=None, color_order='RGB', registered_to_depth, observed_mask=None, hand_visibility=('visible', 'visible'), diagnostics=False)
```

### DepthSurfaceDetector call signatures

```python
DepthSurfaceDetector.detect(depth, object_mask, *, depth_scale_m=0.001, intrinsics=None, table=None, reference_depth=None, observed_mask=None, reference_valid=None, exclude_mask=None, diagnostics=False)
```

### OverheadBoundaryDetector call signatures

```python
OverheadBoundaryDetector.detect(color, depth=None, *, color_order, registered_to_depth=False, depth_scale_m=0.001, object_mask=None, background_color=None, table_mask=None, object_roi=None, depth_roi=None, intrinsics=None, observed_mask=None, diagnostics=False, background_valid_mask=None, background_check_mask=None, background_shift_xy=(0, 0), object_color_seed=None)
```

### Numeric helpers and calibration controls

`background_difference(depth, background, *, depth_unit, background_unit,
invalid_values=())` returns signed float64 background-current in the declared matching input unit
plus validity; both units must be explicitly given and equal (m or mm). `positive_difference_16uc1(depth, background, *,
invalid_values=())` safely casts before subtraction/clipping and returns uint16
positive difference plus validity. Background0 is missing: preserve that returned
validity instead of interpreting every clipped zero as measured background.
`depth_pair_16uc1_to_meters(depth, difference, *, depth_scale_m=.001)` converts the
pair to metres without modifying inputs; it cannot reconstruct lost background validity.
`invalid_values` adds raw sentinels, not thresholds. No automatic scale estimation.

`analyze_color_geometry(detector, depth, colors, hand_rois=None, *, depth_unit,
arm_mask=None, observed_mask=None, work_roi=None,
hand_visibility=('visible','visible'), down=(0,1), depth_jump_m=.1,
diagnostics=False)` retains independent depth/appearance results. Supply a
`ColorAssistResult` and consistent masks; it is not a replacement for measured contact.
An optional `depth_geometry` result can be supplied to reuse matching geometry;
only reuse results from the same input/configuration.

## Debug API: arrays first, explicit PNG saving

There are thirteen renderable spatial result types: `BridgeDrapeResult`, `GraspResult`,
`ObjectOnlyGraspResult`, `LiftResult`, `PlaneFitResult`, `ColorAssistResult`,
`GripperReference`, `ReferenceSegmentationResult`, `SurfaceResult`, `CornerResult`,
`BoundaryResult`, `OuterContourResult` and `BoundaryProximityResult`. CameraIntrinsics/TablePlane and numeric tuples have no renderer;
render their downstream result. Detection and rendering never write files.

All except Corner/Boundary/OuterContour/BoundaryProximity use this signature:

```python
result.render_debug(color=None, *, color_order='RGB', depth=None,
                    depth_scale_m=.001, mask_alpha=.35, panel=True, max_lines=32)
result.save_debug(path, color=None, *, color_order='RGB', overwrite=False, **options)
```

`save_debug` options are the corresponding render options. Corner specializes it:

```python
corner.render_debug(color=None, *, color_order='RGB', depth=None,
                    depth_scale_m=.001, mask_alpha=.2, show_contours=True,
                    max_labels=12, panel=True)
corner.save_debug(path, color=None, *, color_order='RGB', overwrite=False,
                  depth=None, depth_scale_m=.001, **render_options)
```

Boundary adds labels while forwarding common render options:

```python
boundary.render_debug(color=None, *, color_order='RGB', depth=None,
                      depth_scale_m=.001, topology_labels=False, **options)
boundary.save_debug(path, color=None, *, color_order='RGB', overwrite=False, **options)
```

`render_debug` returns a uint8 HxWx3 image in the requested RGB/BGR order, with
potentially larger panel/tile dimensions than the input. `save_debug` writes that
image and returns the destination `pathlib.Path`, not the image array. Neither
method mutates sources/results. Alpha,
label counts, topology_labels, panel/max_lines and show_contours are **display-only**.
Depth rendering uses percentile intensity, not metric height; invalid depth is magenta.
Missing diagnostics are marked limited, not fabricated. Known masks/candidates remain
renderable with diagnosticsFalse; provide source color/depth if no spatial arrays exist.

Save only PNG to an existing non-symlink parent directory. Existing destinations are
refused unless `overwrite=True`; explicit replacement is atomic and never follows an
existing destination symlink into its target. Caller pose keys/labels can appear in
panels; use nonsensitive identifiers. `to_dict()` gives scalar/graph summaries, not
all image arrays. Inspect/export the specific arrays separately with NumPy; raw maps
may contain NaN for unavailable pixels. Use `json.dumps(result.to_dict(), allow_nan=False)`
for strict JSON and do not encode raw NaN arrays as JSON values.

### Complete in-memory Boundary example

This uses an analytic empty/current pair, not real segmentation validation. Save to a
new filename in an existing folder. Change the settings dictionary and keep it with
outputs; all dictionary keys here are public constructor settings.

```python
import json
import numpy as np
from depth_image_lens import OverheadBoundaryDetector

empty = np.full((96, 128, 3), (140, 150, 140), dtype=np.uint8)
current = empty.copy()
current[20:76, 25:103] = (190, 60, 50)
current[48:50, 30:98] = (95, 30, 25)
check = np.zeros((96, 128), dtype=bool)
check[4:14, 4:24] = True  # Known empty table in both analytic images.
settings = dict(background_difference_mode='separate_tolerances',
                background_chroma_tolerance=30., background_luma_tolerance=60.,
                trace_boundaries=True, color_weak_ratio=.7, max_gap_px=0)
model = OverheadBoundaryDetector.from_config(settings)
result = model.detect(current, color_order='RGB', background_color=empty,
                      background_check_mask=check, diagnostics=True)
print(json.dumps(result.to_dict(), allow_nan=False))
image_rgb = result.render_debug(current, topology_labels=True)
result.save_debug('debug-output/boundary-tuning.png', current, topology_labels=True)
# result.parameters contains effective settings, including omitted defaults.
```

For other classes, runnable minimal examples are linked below. These examples generate
synthetic inputs; their camera/reference values are illustrations, not site calibration.

| Feature | Run from repository root |
|---|---|
| Bridge / auto | `python examples/example.py` / `python examples/example_auto.py` |
| Independent grasp | `python examples/example_grasp.py` |
| ObjectOnly / partial gripper metrics | `python examples/example_object_grasp.py` / `python examples/example_object_context.py` |
| Lift / plane fit | `python examples/example_lift.py` |
| Seeded color / geometry | `python examples/example_rgbd.py` |
| Reference segmentation | `python examples/example_reference_grasp.py` |
| Surface | `python examples/example_surface.py` |
| Corner | `python examples/example_corner.py` |
| Boundary / background | `python examples/example_boundary.py` / `python examples/example_boundary_background.py` |
| Topology / seeded shadows / separate tolerances | `python examples/example_boundary_topology.py` / `python examples/example_boundary_shadow.py` / `python examples/example_background_tolerances.py` |

The installed `depth-image-lens --help` describes the **Bridge** NPY/NPZ+JSON command;
it is not a universal CLI for all detectors. See [CLI/input details](DIAGNOSTICS.md).
Dataset-specific tuning runners, scene splits, numerical searches, real-data guides
and galleries are intentionally outside Git. They are not runtime dependencies or
additional public algorithms. No dataset accuracy is implied by these examples.

## OuterContourDetector: ordered external intervals

See [OUTLINE.md](OUTLINE.md) for the complete raw-edge ownership schema, per-call
metric/observation contracts, query settings and runnable example. The mask-only
API composes with segmentation; it does not itself perform color separation.

| Constructor parameter | Default | Unit / increasing-value effect |
|---|---|---|
| `smoothing_px` | `1.0` | Gaussian sigma px; smooths turn noise but may suppress small corners; raw evidence unchanged |
| `corner_window_px` | `8.0` | Arclength px; broader turn support |
| `corner_turn_deg` | `45.0` | Degrees; fewer sharp corner events |
| `min_corner_spacing_px` | `8.0` | Cyclic arclength px; suppresses nearby events |
| `straight_turn_deg` | `5.0` | Degrees; admits more local low-curvature straight proposals |
| `line_error_px` | `1.5` | Maximum raw TLS perpendicular error px; admits noisier/curvier lines |
| `max_arc_chord_ratio` | `1.15` | Ratio; permits more raster roughness/backtracking |
| `min_section_length_px` | `8.0` | Arclength px; marks more short sections unknown |
| `min_straight_length_px` | `20.0` | Arclength px; requires more straight support |
| `clip_margin_px` | `1` | Integer px; widens frame/unobserved-domain uncertainty |
| `max_depth_jump_m` | `0.02` | Camera-Z metres; tolerates larger adjacent depth changes for complete metric length |

`detect(object_mask, *, observed_mask=None, depth=None, depth_scale_m=.001,
intrinsics=None, registered_to_mask=False, depth_valid_mask=None, diagnostics=False)`
returns an OuterContourResult. Missing/bad optional metric inputs preserve pixel
geometry. Explicit domain/mask contract errors remain fatal. No target length is
used to construct predictions. `query_edge_length(target_length, *, unit='px',
tolerance=None, mode='match', min_support_px=20., max_line_error_px=None,
max_metric_line_error_m=.005, require_corner_endpoints=False)` is a separate result query. ToleranceNone means
5px or0.005m; max_line_error_pxNone inherits the detector. The separate query-only
3D line-error limit defaults to0.005m; image-straight does not imply straight in3D.
See the detailed guide
for match versus at_least and the limits of exposed-material-edge interpretation.

`render_debug(color=None, *, color_order='RGB', depth=None, depth_scale_m=.001,
show_raw=True, show_ids=True, panel=True)` returns uint8; `save_debug` uses the same
explicit PNG/non-overwrite contract and returns Path. Rendering flags affect display
only. Run `python examples/example_outline.py --output NEW_IMAGE.png`.

## BoundaryResult outer-proximity selection

`result.filter_outer_proximity(*, near_px=8., reject_fraction=.7,
include_holes=False, observed_mask=None)` selects topology candidates without
changing source masks/graphs. [Full contract](BOUNDARY_PROXIMITY.md).

| Parameter | Default | Unit and effect |
|---|---|---|
| `near_px` | 8.0 | Nonnegative px; larger includes more observed samples as near (distance <= limit), and expands artificial-border uncertainty |
| `reject_fraction` | 0.7 | Ratio in (0,1]; lower excludes more paths (near fraction >= limit) |
| `include_holes` | False | True also treats background holes as boundaries; False fills holes per component |
| `observed_mask` | None | Optional bool HxW known observed domain; missing provenance cannot reveal an artificial internal crop |

The result's kept/excluded/pending/ignored lists retain IDs and copied measurements.
Only unique strong/weak coordinates contribute statistics; inferred samples do not.
Frame/known ROI uncertainty is pending, not a physical exterior decision. Short paths
stay ignored. Distance is to nearest background pixel center, not contour centers.
`render_debug(color=None, *, color_order='RGB', depth=None, depth_scale_m=.001,
show_ids=False, panel=True)` returns a before/after uint8 image; inherited save_debug
returns a Path with explicit non-overwrite PNG saving. No mask_alpha/max_lines options.
Run `python examples/example_boundary_proximity.py --output NEW_IMAGE.png`.

### Outer contour size and ordered traversal

These are result helpers, not detection thresholds. `result.largest_contour(*,
by='area_px2')` selects enclosed polygon area by default; alternatives are
`foreground_pixel_count` (holes excluded) and `perimeter_px`. Ties use the smallest
ID; empty results return None. `result.resolve_sequence(contour_id=None)` resolves
stored corner/segment IDs to detached records in cyclic order; None selects the
largest area, empty returns [], invalid/duplicate/missing references raise ValueError.
See [size and sequence semantics](OUTLINE.md#contour-size-largest-selection-and-sequence-traversal).
The public outline example demonstrates both helpers. Sizes and selection do not
change existing detection/query decisions; pixel area and physical area differ.
