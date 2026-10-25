# depth-image-lens

Experimental shape heuristics (NumPy-only core) for a suspended **deformable object** in static depth images. `BridgeDrapeDetector` estimates a bridge between two candidate anchors, large apparent sag, and left/right hanging extent. It does not verify physical holding, tension, or metric height above a table.

The repository contains the implementation. No ZIP download is required.

## Quickstart: two in-memory uint16 arrays

If your preprocessing already provides 2D NumPy ndarrays, no ROS or file conversion is needed:

```python
from depth_image_lens import BridgeDrapeDetector

# current_depth and masked_difference: matching HxW np.uint16 arrays.
result = BridgeDrapeDetector(threshold_m=0.02).detect_16uc1(
    current_depth, masked_difference,
    depth_scale_m=0.001,  # Explicit convenience default: 1 count = 1 mm.
    upper_band=(8, 20),  # Adjust to the hand-to-object region in your image.
)
print(result.to_dict())
```

Here `masked_difference = max(empty_background - current_depth, 0)` before any additional region exclusion. Finite difference zero means observed background/exclusion; current-depth zero means missing depth. The positive values highlight surfaces closer to the camera than the empty background. `16UC1` specifies a single-channel unsigned-16 type, **not a distance unit**: change `depth_scale_m` for a sensor with a different metres-per-count scale. The same scale applies to both arrays. Inputs are not modified, and noncontiguous views are accepted. Float conversion allocates separate metric buffers.

For manual anchors, pass `hand_rois={"left": (...), "right": (...)}`. For missing background or visibility, also pass the appropriate bool `background_valid`/`observed_mask`; zero alone cannot preserve that provenance. The general `detect()`/`detect_auto()` APIs still require explicit m/mm units and never infer them from dtype.

## Independent table-relative lift

`LiftDetector` measures signed distance from a caller-supplied table plane. It needs rectified optical-axis depth, depth-camera intrinsics, an oriented plane in that same camera frame, and a deformable-object mask. It does **not** need hand ROIs, a difference image, or world/robot extrinsics. These calibration inputs are unnecessary for the image-relative bridge/sag detector above.

```python
from depth_image_lens import CameraIntrinsics, TablePlane, LiftDetector

# Supply actual camera parameters and plane coefficients; do not copy a sample calibration.
intrinsics = CameraIntrinsics(fx, fy, cx, cy)
table = TablePlane((nx, ny, nz), offset_m)
result = LiftDetector().detect_16uc1(
    current_depth, intrinsics, table, object_mask,
    depth_scale_m=0.001, whole_object_visible=False,
)
print(result.to_dict())
```

`state` describes visible-region elevation (`lifted`, `not_lifted`, `unknown`). `whole_object_lifted` is separate: visible elevation alone cannot prove the entire object is off the table. An observed near-table component can give whole-object `no` even in a partial view. This is geometric evidence, not proof of grasp success. Masks and intrinsics are caller inputs. The optional `fit_table_plane` helper fits a plane from a manually selected empty-table ROI; it does not perform semantic segmentation. See [the lift contract](API.md#table-relative-lift) and run `python examples/example_lift.py` for a synthetic oblique-plane example.

To obtain the plane from an empty image rather than entering its coefficients:

```python
from depth_image_lens import fit_table_plane

fit = fit_table_plane(empty_depth, intrinsics, table_mask, depth_scale_m=0.001)
print(fit.to_dict())  # Inspect point count, metric residuals and warnings.
result = LiftDetector().detect_16uc1(current_depth, intrinsics, fit.plane, object_mask)
```

The same explicit scale must be used for fit and detection. For float arrays already in metres, pass `depth_scale_m=1.0` to the fit and use `LiftDetector.detect`. The default orientation assumes the camera origin is on the above-table side. The ROI must contain only the empty table; clutter or a wall can yield a wrong plane even with small residuals. Refit after camera/table changes. See [fit details](API.md#empty-table-plane-helper).

## Install

Python 3.10 or newer and NumPy 1.24 or newer are required. From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The distribution name is `depth-image-lens`; the import name is `depth_image_lens`:

```python
from depth_image_lens import BridgeDrapeDetector, BridgeDrapeResult
```

If NumPy, setuptools >=64 and wheel are already available in the active environment, an offline alternative is:

```bash
python -m pip install --no-index --no-build-isolation --no-deps -e .
```

The standard command uses build isolation and may download declared build dependencies. Tests use the standard-library `unittest` module; no extra test dependencies are required. On systems without `venv`/`ensurepip`, use an existing environment manager to create a virtual environment with pip before installing. Do not install into the system Python.

## Run

After installation, from the repository root:

```bash
python examples/example.py
python examples/example_auto.py
python -m unittest discover -s tests -t . -v
```

The two examples generate a synthetic U-shaped bridge and print JSON. To supply aligned arrays and configuration:

```bash
python examples/example.py depth.npy difference.npy examples/config.example.json
python examples/example_auto.py depth.npy difference.npy examples/config.auto.json
```

NPY loading disables pickle. Adjust units, masks, ROIs and thresholds for your own inputs. Execution makes no network requests or robot commands.

## Use your own depth pair

Keep private input files in the ignored `data/` directory (or outside the repository). These must be numeric depth arrays, not colored depth visualizations. Use a fixed camera, unchanged empty background, and identical registered pixels. Choose explicit `m` or `mm`; units and registration are not inferred.

For current depth plus empty background, edit `examples/config.background.json` for your units, upper band and actual sensor sentinel values, then run:

```bash
depth-image-lens --depth data/current.npy --background data/empty.npy --config examples/config.background.json --output debug-output
```

For a precomputed difference, keep explicit `difference_unit` and `difference_sign` in the configuration:

```bash
depth-image-lens --depth data/current.npy --difference data/difference.npy --config examples/config.auto.json --output debug-output
```

A single archive is also supported: `np.savez("data/pair.npz", depth=current, background=empty)` or `np.savez("data/pair.npz", depth=current, difference=difference)`. Optional NPZ bool-mask keys are `background_valid`, `observed_mask`, `arm_mask` and `work_roi`.

```bash
depth-image-lens --input data/pair.npz --config examples/config.background.json --output debug-output
```

The CLI prints strict JSON and saves `result.json`, `arrays.npz` and `overlay.svg`. It always requests diagnostics. Set `"mode": "manual"` and add `hand_rois` for manual mode; the default is automatic candidates. Config mask paths are relative to the config file. Pickle/object arrays and nonfinite JSON constants are rejected. No input paths or user metadata are copied into the result or overlay. `python -m depth_image_lens.cli` is an equivalent entrypoint.

Tune units/invalid values first, then the upper band and arm exclusion, foreground threshold, and sag/drape ratios. The synthetic U example returns yes/yes/both. Missing opposite evidence, clipping, invalid depth and excessive depth jumps can correctly return unknown; inspect [diagnostics](DIAGNOSTICS.md) before relaxing thresholds. A helper-generated difference is not evidence of real-image accuracy. No PNG/RGB-to-distance conversion is provided.

## Masked positive-foreground differences

If unrelated regions have already been set to zero, use `difference_mode="positive_foreground"` with precomputed difference input:

```bash
depth-image-lens --depth data/current.npy --difference data/masked_difference.npy --config examples/config.positive.json --output debug-output
```

In this mode, a finite zero difference is **known background/exclusion**, not missing depth. Positive values above `threshold_m` are candidate foreground; zero is never foreground even with threshold zero, and negative values are excluded. Raw depth <=0/NaN, nonfinite difference, or false `observed_mask`/`background_valid` still means unknown. An all-zero difference with valid observed raw depth can return no foreground instead of unknown.

The supplied positive values are used as-is. No sign inversion or physical background reconstruction is performed. The input and threshold must still have declared matching depth units. Do not convert missing measurements to zero: preserve NaN or the appropriate validity mask. The `features.difference_mode` output records which contract was used.

The default `signed` mode and the optional `background_difference()` helper remain unchanged: they preserve explicit `background-current` / `current-background` handling and validate the implied physical background. These are different input contracts; only explicit `encoding="16UC1"` opts a raw background pair into clipped-positive preparation.

## Inputs and outputs

- Two matching arrays: current depth and either a signed physical difference or a masked positive-foreground difference. Select the input contract explicitly; see below.
- Manual mode: caller-provided hand search ROIs. Automatic candidate mode: a coarse upper search band, with optional arm exclusion. Neither mode recognizes hands semantically.
- Independent outputs: `bridge=yes/no/unknown`, `large_sag=yes/no/unknown`, and `drape=left/right/both/neither/unknown`, plus features, masks and reasons.
- Intrinsics and world/robot calibration are not required for these image-relative estimates. Metric depth units are still required for depth thresholds.

See [API and tuning](API.md), [bridge-first candidates and comparison](AUTO.md), and [diagnostic arrays and SVG overlays](DIAGNOSTICS.md).

## Layout

```text
pyproject.toml
src/depth_image_lens/  # single implementation and public exports
examples/             # manual/automatic examples and configuration
tests/               # synthetic regression and comparison
API.md / AUTO.md
LICENSE               # MIT
bridge_drape.py       # minimal legacy import wrapper; install first
```

The source module is canonical; the root compatibility wrapper contains no detection logic. Local environments, historical archives and machine-specific records are ignored by Git.

## Evidence and limitations

Focused synthetic unit tests exercise API behavior, missingness and known
limitations. They are not independent real observations or field-accuracy evidence.

An arm shaped exactly like a bridge can produce the same depth evidence as a deformable object. An exclusion mask or other semantic input is required in that case. Unknown results include some true bridges. No matched real empty-background/current-depth pairs have been evaluated; real-image accuracy is unvalidated.

## License

Original project code is licensed under [MIT](LICENSE). NumPy is an external dependency under its own BSD-3-Clause license; it is not bundled here. Build tools keep their own licenses. No research images, datasets, third-party source copies or credentials are distributed in this repository. This inventory is not a legal guarantee concerning every possible downstream use.

Tune units and validity first, then geometry ROIs and thresholds. Pixel lengths
scale with image resolution, areas quadratically; metric thresholds and dimensionless
ratios do not. Keep benchmarks, comparison runners and local HTML reviews outside
the repository. No optimum or field accuracy is inferred from parameter adjustment.

For optional file-based verification of the same uint16 contract:

```bash
depth-image-lens --depth data/current.npy --difference data/masked_difference.npy --config examples/config.16uc1.json --output debug-output
```

This configuration declares `encoding="16UC1"` and one `depth_scale_m` (default 0.001). Do not mix this convenience setting with separate unit/sign settings. With `--background` instead, the CLI safely creates clipped uint16 differences and retains the validity mask before detection.

Bridge continuity tuning: manual and automatic modes now share `depth_jump_m`. Optional tiny-gap inference additionally checks raw depth through the gap and records inferred pixels separately from observed foreground. Inspect candidate decisions and the yellow/pink debug overlays before changing gap or depth thresholds; see [API details](API.md#bridge-depth-continuity-and-explicit-gap-inference). Automatic selection can continue to a second eligible candidate after rejecting the first discontinuous one.


For close views with missing hands or the upper third cropped away, see the
[close-view contract](AUTO.md#close-views-missing-hands-and-the-upper-third).
Visible connection candidates can be used without full hand/arm visibility;
clipped supports cannot be treated as absent or assigned a physical hand ID.
`result.stage_results` keeps geometry measurements, inferred static support
hypotheses and evidence warnings independently. See
[feature-wise continuation](RGBD.md#independent-features-and-evidence-quality).


## Optional registered color assistance

`SeededColorAssistant` accepts the same uint16 depth/difference arrays plus
registered RGB/BGR color and caller-provided object/hand seed masks. It returns
appearance candidates and unknown pixels, retaining the independent depth-only
result. See [RGBD.md](RGBD.md) for ndarray inputs, the small integration example,
the input contract and important real-image limitations. It adds no
runtime dependency and does not establish physical grasp or hand identity.

The RGB-D example keeps sensor validity separate from appearance uncertainty;
optional route checks do not replace the independent depth-only result.

## Optional overhead corner candidates

`OverheadCornerDetector` accepts registered RGB/BGR and optical-axis depth, with
a supplied object mask or matched empty-scene color. It compares polygon angles,
local line fits and local depth checks. Accepted corners are not proof of a single
layer or safe grasping. The optional `corner` extra uses OpenCV (Apache-2.0); the
base package still needs only NumPy. See [CORNER.md](CORNER.md) for the ndarray
API, runnable example and limitations. Field improvement is not established.
External research data and evaluation outputs are not distributed in this repo.

Corner support refinement is available as `support_policy="adaptive"`; fixed
support remains the default. See [the bounded support contract](CORNER.md#bounded-adaptive-support-opt-in).

## Repository scope

Keep library code, focused unit tests with small synthetic fixtures, and minimal
API examples here. Evaluation runners, parameter grids, timing benchmarks, dataset
intake/annotation tools, detailed research reports and HTML/image galleries belong
in a separate local evaluation directory, not Git. Review generated HTML locally;
no upload is required. Historical evaluation copies are preserved externally and
Git history is not rewritten. Public tests and examples need no external evaluation
files. `examples/example_rgbd.py` retains reusable integration helpers and a small
ndarray example, not a benchmark runner.

## Independent image-side grasp estimates

`IndependentGraspDetector` uses separated hand/object masks and local depth
consistency to return left/right visual grasp hypotheses independently, without
bridge or sag prerequisites. Physical grasp remains unverified. See
[the simple-mode contract](GRASP.md) and `examples/example_grasp.py`.


Corner proposal coverage can be explored with the optional
`proposal_policy="multiscale"`; the existing `"single"` default remains available.
See [multiscale exterior proposals](CORNER.md#optional-multiscale-exterior-proposals)
for epsilon scaling, deduplication and limitations.


Corner tuning has an explicit opt-in starting preset:
`OverheadCornerDetector.recommended_for_tuning()`. It uses multiscale proposals,
bounded adaptive support and diagnostic-only long/short angle difference. Ordinary
constructor defaults stay compatible. Independent proposal/fitted angle,
intersection shift, minimum support and internal-step/boundary thresholds are
listed in [CORNER.md](CORNER.md#explicit-recommended-starting-preset). This is a
starting point for real-data tuning, not a calibrated confidence score or a
measurement of table-relative height or usable grasp area.


## Object-only inferred holding

`ObjectOnlyGraspDetector` needs no visible hands: under an explicit static
suspension rule, sufficient reference elevation implies inferred holding and
sufficient same-component visible width implies inferred bilateral support.
Use a selected object mask plus current depth and either an empty-background
difference or a calibrated table plane. A missing reference remains unknown.
These are configurable shape rules, not physical grasp verification; existing
`IndependentGraspDetector` behavior is unchanged. See [OBJECT_GRASP.md](OBJECT_GRASP.md)
and `python examples/example_object_grasp.py`.

Corner results also provide `result.render_debug(color, color_order="RGB")` and
explicit `result.save_debug("corner-debug.png", color, color_order="RGB")`.
See [Corner rendering and input contracts](CORNER.md#render-and-save-debug-images)
and `examples/example_corner_debug.py`. Detection itself never writes files.
