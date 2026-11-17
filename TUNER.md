# Local parameter tuning

The tuner works on saved, already registered images. It does not acquire camera
frames, register RGB to depth, command a robot, or upload a dataset. Its output is
an experimental configuration, not a guarantee of segmentation or grasp safety.

## Install and launch

Install the optional GUI dependencies, then start the Qt Widgets application:

```sh
python -m pip install -e ".[gui]"
depth-image-lens-tuner
# Equivalent:
python -m depth_image_lens.tuner
```

The GUI extra uses **PySide6 Essentials** and headless OpenCV. Browser rendering,
a web server, Qt WebEngine and a network service are not required. Importing the
core library does not import Qt.

## Quick workflow

1. Select current `color` and `depth`, plus the empty-scene `background_color`
   and `background_depth`. For an externally supplied mask, select `mask` as the
   source and choose a single-channel mask file.
2. Set the crop and depth unit. Choose a segmenter mode, optional repair, and
   target (`mask`, `count`, or `outline`).
3. Select a stage and parameter. Enter a JSON value (`true`, `false`, `null`, a
   number or a string), or adjust the numeric slider. Its displayed minimum and
   maximum can be edited. `Edit all settings` edits the full configuration.
4. Click **Run**, or leave **Auto-run after edit** enabled. Use the view selector
   to inspect the overlay, individual masks, debug composites and recorded maps.
   Switching a view does not rerun detection. With Full diagnostics enabled,
   select a `model_all_*` map and its model ID, or `model_rejection_panels`
   for that model's separate rejection-reason tiles. These views are rendered
   on selection, not as a pre-rendered image for every model. **Fit** scales the display only;
   clearing it shows native image pixels with scrolling.
5. **Save project** records input paths, crop and parameters in JSON. **Load
   project** restores them. **Export result** writes a new subdirectory containing
   the result's configuration snapshot, summary, raw arrays, PNG views and their
   name index. Exported paths can identify local files; review a project before
   sharing it externally.

### Responsiveness and cancellation

Detection runs in one spawned worker process, not in the Qt event loop. Slider
dragging updates only the displayed value; releasing it commits the parameter.
Auto-run debounces committed edits for 250 ms. There is at most one running job
and one replaceable pending request. Older results and errors are discarded once
settings change; the GUI does not queue every intermediate slider position.

**Cancel** discards the displayed outcome of the active job and clears pending
work, including any edit-debounce timer. Turning Auto-run off also cancels its
pending debounce. A running detector call is not force-terminated. Batch and search jobs
check cancellation between samples/trials. Exports already started continue to
write their new output directory, even if their completion display is cancelled.
Closing the window can therefore leave the worker finishing its current call.
These choices avoid corrupting an in-progress native-library call or file write.

## Input grid and units

Each supplied array must be **640 x 400** or **1280 x 800**. These resolutions
may be mixed: for example, 1280 x 800 color with 640 x 400 depth. Current and
background color/depth, masks and ground truth are normalized independently.
All inputs must already describe the same registered field of view.
Color arrays are RGB `uint8`. Color files are decoded to RGB. Depth is optical-axis
camera Z, not distance along a ray, and requires a declared `depth_scale_m`:
`0.001` for integer millimetres, `1.0` for metre values.

Preprocessing has a fixed order:

1. Normalize each input independently to 640 x 400. Color uses area reduction. Depth and
   masks use nearest pixel-centre samples, preserving stored values and invalid
   depth values rather than averaging across an edge.
2. Apply `crop = [x, y, width, height]` in that normalized grid. The bounds are
   `[x, x + width)` and `[y, y + height)`. Both dimensions must be at least two
   pixels, and the crop must fit entirely in 640 x 400.
3. Apply the selected pipeline to the cropped arrays. Pixel-radius and area
   parameters therefore refer to the normalized, cropped grid, regardless of
   source resolution.

Image resizing never multiplies or divides the **depth values**. Input images
must already have matching views and alignment; matching array shapes cannot
verify registration. Resampling also cannot create missing sensor evidence.

Output coordinates use the cropped grid. The saved transform records source
shape, scale and crop for each input role. `PreparedInput.to_source_xy(points,
role="color")` maps pixel centres to that original image. Without a role it uses
the current depth grid when available, otherwise color (then a background or the
first available input). Camera intrinsics must describe that same default source
grid, normally the current depth image, not the larger color image. They are
scaled using the same pixel-centre convention, then adjusted for the crop.
This is resampling only: unregistered sensors or different crops/fields of view
need calibration and registration before loading. Matching dimensions cannot
verify alignment.

## Pipeline and interpretation

The source is either `RGBDBackgroundSegmenter` or an externally provided mask.
The segmenter supports `union`, `depth_required`, `evidence`, `color_seeded`,
`depth_anchored`, `color_supported`, `color_region`, `color_local` and `color_model`.
The optional repair step uses `cleanup_mask`. The target is the mask itself,
`OverheadObjectCounter`, or `OuterContourDetector`.

- Segmenter input requires current color/depth and empty-scene color/depth.
- External-mask input still requires current color for display and color-based
  counting. A mask is interpreted as nonzero foreground, not an instance label map.
- Counter depth inputs must be supplied as a current/background pair. Without
  them, candidate groups may be available but an accepted count is not guaranteed.
- Upstream missingness, uncertainty and repair changes are passed to downstream
  counting. Repair additions are geometric hypotheses, not new depth observations.
- `result/mask` is the pipeline mask before counter-specific filtering. Inspect
  the counter's candidate, rejected, excluded and group views separately.
- An ambiguous count is not replaced by `candidate_count`. Candidate groups are
  hypotheses; the reasons and warnings remain part of the result.
- Outline distances are in pixels unless metric inputs are available. Inspect
  the outline summary for metric-input availability; never interpret pixel
  lengths as millimetres.

## Debug views

Standard results include the source arrays, pipeline mask and overlay, available
result masks, and each class's existing composite debug image. Detailed recording
adds intermediate decision maps and scalar arrays. This does not change the
intended detection result, but can substantially increase runtime and memory.

In particular, `color_model` can record per-model stacks. Merely displaying one
panel does **not** mean only that panel was calculated. Enable detailed diagnostics
only when needed. Common composite images may show only a subset of masks; use
the individual selectable maps for other available fields.

Scalar previews use each map's finite minimum and maximum for display. A brighter
pixel in one image cannot be compared directly to a brighter pixel in another
without checking raw values and units. Raw arrays should be used for quantitative
analysis, not colors sampled from the rendered preview.

## Automatic evaluation

Ground truth is never passed to the detector as an input. Mask tuning uses
intersection-over-union (IoU); two empty masks score 1. Count tuning compares the
accepted count to the expected integer count. An ambiguous `count=None` does not
silently become a correct candidate count.

A search varies an explicit finite list of values for each `stage.parameter`.
Grid search visits combinations deterministically; random search uses a recorded
seed. The trial budget bounds the search. Invalid parameter combinations are
reported as failed trials instead of silently being accepted.

Select settings using the training samples, then evaluate the winning setting
on a separate validation set. Sample IDs must be unique and training/validation
IDs must not overlap. This is a bookkeeping check, not a proof of independence:
do not give copies or near-identical captures of the same scene different IDs
and put them in both sets. Prefer held-out object placements, lighting and shadow
conditions. A single good score cannot establish robustness to unseen scenes.

## Batch manifest and search controls

Use **Batch / search**, choose a local JSON manifest and select `batch`, `grid`
or `random`. The current crop and pipeline configuration apply to every sample.
The search dialog accepts the objective, maximum trials, random seed and JSON
search space. For example:

```json
{
  "segmenter.background_chroma_tolerance": [15, 20, 25],
  "segmenter.background_luma_tolerance": [40, 60]
}
```

A manifest uses paths relative to its directory (absolute paths also work):

```json
{
  "version": 1,
  "samples": [
    {
      "id": "train-placement-01",
      "split": "train",
      "arrays": {
        "color": "train/color.png",
        "depth": "train/depth.png",
        "background_color": "empty/color.png",
        "background_depth": "empty/depth.png",
        "ground_truth_mask": "train/truth.png"
      },
      "expected_count": 2
    },
    {
      "id": "held-out-placement-02",
      "split": "validation",
      "arrays": {
        "color": "validation/color.png",
        "depth": "validation/depth.png",
        "background_color": "empty/color.png",
        "background_depth": "empty/depth.png",
        "ground_truth_mask": "validation/truth.png"
      },
      "expected_count": 1
    }
  ]
}
```

`ground_truth_mask` is required only for mask-IoU search; `expected_count` is
required for count search and must be a nonnegative integer. Choose the `count`
target for that objective. Plain batch evaluation can run without labels and
reports measurements without claiming an accuracy score.

For an NPZ containing multiple arrays, a manifest entry can select a key:
`"depth": {"path": "capture.npz", "key": "depth_mm"}`. NumPy loading disables
pickle. Masks are boolean/nonzero arrays. Depth files must be lossless PNG, NPY
or NPZ; never use JPEG for depth. Manifest samples can additionally supply
`observed_mask` and `exclusion_mask`; the latter currently applies to counter
filtering only. A sample's optional `intrinsics` object has default-source-grid
`fx`, `fy`, `cx` and `cy` for metric outline measurements (the current depth grid
when supplied, as described above).

Results appear as a structured summary. **Apply best** explicitly applies the
winning training configuration; validation results do not choose the winner.
The prior preview image is not replaced by an unrelated sample during batch
execution. Run the desired sample again to inspect its new settings visually.

## Current limits

- There is no live-camera mode or robot integration.
- Label drawing and semantic instance annotation are not provided. Prepare mask
  labels externally; the workbench loads them for evaluation.
- The GUI shows one selected debug image rather than an always-rendered gallery.
  Pixel-click inspection is available; side-by-side comparison is not yet provided.
- Each committed run recomputes the pipeline; preceding-stage caching is not yet
  implemented. The latest-only scheduler prevents a backlog, not computation time.
- Full per-model diagnostics are opt-in and can be expensive; viewing one image
  does not reduce the amount recorded by an existing class.
- The GUI input buttons cover the common five inputs. Additional observation and
  exclusion masks, explicit NPZ keys and intrinsics are available through the
  batch manifest/backend.
- Search is a bounded grid/random comparison, not a guarantee of a global optimum.
  It currently scores mask IoU or accepted-count correctness, not corner accuracy.
- Synthetic tests and an offscreen Qt smoke test do not establish real-camera
  accuracy or identical responsiveness on every machine.

## Detailed segmentation diagnosis

Enable **Full diagnostics** to record independent gate maps in every fusion mode.
Select a mask in the view list and enable **Mask overlay** to highlight it on the
original color image. Click the displayed image to inspect its raw value, crop
coordinates, full 640 x 400 coordinates, and all active reason masks at that
pixel. Descriptions, units and thresholds appear in the view legend where
available. Scalar maps are display-normalized, not probabilities; compare raw
values or gate margins rather than brightness across images.

- `color_region`: inspect region/color-group IDs, assigned representative color
  distance, region area, valid-depth fraction, clipped mean depth and positive
  depth fraction. Each region gate has its own pass/fail mask. Region acceptance
  is separate from per-pixel invalid-depth/farther veto and final area removal.
  Small area and weak depth may fail simultaneously. Unassigned candidates are
  shown by the color-budget mask, not silently treated as failed region tests.
- `color_local`: separate coverage, effective sample count, mean-depth, density,
  seed-area and table-like gates. Inspect final growth-boundary source-color and
  neighboring-color distances and their independent rejection/next-step masks.
- `color_supported`: inspect raw/retained seeds, growth domain, color boundary,
  background guard and the later measured-depth-fraction component veto.
- `color_seeded`: inspect strict chroma seed margin, local depth support,
  relaxed-color domain, growth steps and domain pixels not selected.
- `depth_anchored`: inspect anchors, negative/background evidence, refinement
  domain, score and score threshold failure.
- `evidence`, `union`, `depth_required`: inspect raw depth/color gate margins,
  modality availability, uncertainty, final area removal and applicable fusion
  decision masks. Evidence mode also exposes neighboring depth support.

A `gate_*_margin` is value minus threshold; positive passes, and equality follows
its displayed comparison (`>` or `>=`). NaN denotes an out-of-scope/undefined
measurement, shown magenta. A failed seed condition is not necessarily a final
rejection: subsequent growth or a model override can still select that pixel.
Reason masks overlap and describe observed conditions, not a counterfactual
claim that changing one parameter will fix the result. Growth-boundary distances
are independently minimized over selected neighbors; their minima need not come
from the same neighbor. The paired color-failure mask uses both tests on the
same neighbor. A compatible next-step mask is a budget diagnostic, not an
instruction to fill the whole remaining domain.

These diagnostics are generated by the core segmenter, not reimplemented in the
GUI. Recording is opt-in and never feeds back into selection. Local/supported
boundary inspection performs eight additional neighbor passes only when enabled;
full diagnostics therefore cost extra memory and time. Display switching and
pixel inspection reuse stored results and do not rerun detection.

The optional `color_model` setting `model_training_policy: "color_core"` enables
training-only interior validation. See `RGBD_COLOR_MODEL.md` for limitations.
The `model_training_input_mask` and `model_training_used_mask` panels show actual
palette proposal/fit input, separately from the legacy final-selected training
mask. Independent training failure panels and group statistics explain rejected
colors without treating overlapping conditions as unique causal reasons.
