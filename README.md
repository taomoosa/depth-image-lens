# depth-image-lens

Experimental shape heuristics (NumPy-only core) for a suspended **deformable object** in static depth images. `BridgeDrapeDetector` estimates a bridge between two candidate anchors, large apparent sag, and left/right hanging extent. It does not verify physical holding, tension, or metric height above a table.

For site-specific adjustment, start with the [parameter and debug reference](TUNING.md):
detector settings, units, interactions, uncertainty and explicit PNG output.
Defaults are provisional; the guide distinguishes image evidence from physical claims.

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

Tune units/invalid values first, then the upper band and arm exclusion, foreground threshold, and sag/drape ratios. The synthetic U example returns yes/yes/both. Missing opposite evidence, clipping, invalid depth and excessive depth jumps can correctly return unknown; inspect [diagnostics](DEBUGGING.md#bridge-drape-diagnostics) before relaxing thresholds. A helper-generated difference is not evidence of real-image accuracy. No PNG/RGB-to-distance conversion is provided.

## Masked positive-foreground differences

If unrelated regions have already been set to zero, use `difference_mode="positive_foreground"` with precomputed difference input:

```bash
depth-image-lens --depth data/current.npy --difference data/masked_difference.npy --config examples/config.positive.json --output debug-output
```

In this mode, a finite zero difference is **known background/exclusion**, not missing depth. Positive values above `threshold_m` are candidate foreground; zero is never foreground even with threshold zero, and negative values are excluded. Raw depth <=0/NaN, nonfinite difference, or false `observed_mask`/`background_valid` still means unknown. An all-zero difference with valid observed raw depth can return no foreground instead of unknown.

The supplied positive values are used as-is. No sign inversion or physical background reconstruction is performed. The input and threshold must still have declared matching depth units. Do not convert missing measurements to zero: preserve NaN or the appropriate validity mask. The `features.difference_mode` output records which contract was used.

The default `signed` mode and the optional `background_difference()` helper remain unchanged: they preserve explicit `background-current` / `current-background` handling and validate the implied physical background. These are different input contracts; only explicit `encoding="16UC1"` opts a raw background pair into clipped-positive preparation.

For the file-based uint16 convenience contract:

```bash
depth-image-lens --depth data/current.npy --difference data/masked_difference.npy --config examples/config.16uc1.json --output debug-output
```

This configuration declares `encoding="16UC1"` and one `depth_scale_m` (default
0.001). Do not mix it with separate unit/sign settings. With `--background`, the
CLI creates clipped uint16 differences and retains their validity mask.

## Choose a workflow

The core package requires only NumPy. OpenCV is optional for color-boundary,
corner and outline processing and for PNG debug rendering:

```bash
python -m pip install -e '.[boundary,corner,outline,debug]'
```

Use the guide for the evidence available in your application:

| Task | Guide | Minimal example |
|---|---|---|
| Bridge, apparent sag and hanging extent | [API and input contract](API.md), [automatic anchors and close views](AUTO.md) | `examples/example_auto.py` |
| Table-relative lift and empty-table plane fitting | [Calibrated lift](API.md#table-relative-lift) | `examples/example_lift.py` |
| Seeded color assistance with independent depth evidence | [Registered RGB-D assistance](RGBD.md) | `examples/example_rgbd.py` |
| Local hand/object contact hypotheses | [Independent grasp](GRASP.md) | `examples/example_grasp.py` |
| Static support inference without visible hands | [Object-only inferred holding](OBJECT_GRASP.md) | `examples/example_object_grasp.py` |
| Repeatable-pose gripper references | [Reference segmentation](REFERENCE.md) | `examples/example_reference_grasp.py` |
| Elevated and nonflat surface evidence | [Surface inspection](SURFACE.md) | `examples/example_surface.py` |
| Visible overhead corner candidates | [Corners](CORNER.md) | `examples/example_corner.py` |
| Color boundaries, depth steps and relative-depth islands | [Overhead boundaries](BOUNDARY.md) | `examples/example_boundary.py` |
| Ordered exterior contours and length queries | [Outlines](OUTLINE.md) | `examples/example_outline.py` |
| Open-reference to partial-exterior matching | [Outline matching](OUTLINE_MATCH.md), [corner-based matching](CORNER_OUTLINE_MATCH.md) | `examples/example_outline_match_sources.py` |
| Current/empty RGB-D foreground masks | [Background segmentation](RGBD_BACKGROUND.md) | `examples/example_rgbd_background.py` |
| Visible distinct-color deformable-object counts, from RGB-D or an existing mask | [Object counting](OBJECT_COUNT.md) | `examples/example_object_count.py`, `examples/example_object_count_mask.py` |
| Optional small-hole filling and mask exterior smoothing | [Mask cleanup](MASK_CLEANUP.md) | `examples/example_mask_cleanup.py` |
| Supported overhead pick geometry | [Pick candidates](PICK_CANDIDATES.md) | `examples/example_pick_candidate.py` |
| Raised outer-edge bands | [Edge bulge evidence](EDGE_BULGE.md) | `examples/example_edge_bulge.py` |

Additional boundary guides cover [separate color/brightness tolerances](BACKGROUND_TOLERANCES.md),
[seeded shadow hypotheses](BOUNDARY_SHADOW.md), [continuity and topology](BOUNDARY_TOPOLOGY.md)
and [outer-proximity selection](BOUNDARY_PROXIMITY.md).

For image-column selection, run `python examples/example_vertical_interior.py --x 16`.
The [column-selection contract](OUTLINE.md#select-an-interior-point-in-one-column)
explains exact component masks, visible-run clearance and clipping limitations.

## Tune and inspect results

- [Parameter reference](TUNING.md): defaults, units, interactions and call-time controls.
- [RGB-D/pick tuning](RGBD_PICK_TUNING.md): input checks, settings, symptoms,
  per-pixel evidence and [fixed settings across captures](RGBD_PICK_TUNING.md#fixed-settings-across-captures).
- [Diagnostics and debug output](DEBUGGING.md): arrays, strict JSON, SVG overlays
  and optional PNG rendering. Detection never writes files automatically.
- [Dark shadows and nearer occluders](RGBD_BACKGROUND.md#dark-shadows-and-much-nearer-objects):
  depth-band filtering and its thin-object/missing-depth tradeoffs.

Run `python examples/example_rgbd_pick_tuning.py --help` for the registered
current/empty NPZ workflow. It records settings, masks, contour IDs, representative
points and numeric evidence in a new output directory.

Tune units, registration and validity first, then geometry regions and thresholds.
Pixel lengths scale with image resolution and pixel areas scale quadratically;
metric thresholds and dimensionless ratios do not. Keep uncertainty separate from
measured absence. A lower unknown rate is not proof of better detection.

## Evidence and limitations

The library provides experimental geometric and appearance heuristics, not physical
grasp verification, semantic hand recognition, hidden-layer counts or robot control.
Bridge/sag/drape are image-relative; calibrated table-normal height requires the
separate lift API. Same-sized arrays do not establish registration or synchronization.

Focused synthetic unit tests exercise public API behavior, missingness and known
counterexamples. They do not establish real-image accuracy, sensor calibration,
physical grasp success or throughput. Defaults require validation on independent,
representative captures from the deployment setup.

An arm shaped like a bridge can produce the same depth evidence as a deformable
object. Supply semantic exclusions where needed. Cropping, missing depth, occlusion
and ambiguous appearance can legitimately leave results unknown. Feature guides
state the additional assumptions for each workflow.

## Repository scope

- `src/depth_image_lens/`: canonical implementation and public exports.
- `examples/`: minimal runnable API workflows and configuration examples.
- `tests/`: focused regression tests with synthetic fixtures.
- Root Markdown guides: public usage, input contracts and tuning reference.
- `bridge_drape.py`: compatibility import wrapper; install the package first.

Keep datasets, private captures, benchmark/parameter-search runners, investigation
reports and generated galleries outside the public source tree. Tests and examples
do not require external evaluation files. Local environments and generated outputs
are ignored by Git.

## License

Original project code is licensed under [MIT](LICENSE). NumPy and optional OpenCV
are external dependencies under their own licenses and are not bundled here.
