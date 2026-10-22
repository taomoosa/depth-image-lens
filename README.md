# depth-image-lens

Experimental, NumPy-only shape heuristics for a suspended **deformable object** in static depth images. `BridgeDrapeDetector` estimates a bridge between two candidate anchors, large apparent sag, and left/right hanging extent. It does not verify physical holding, tension, or metric height above a table.

The repository contains the implementation. No ZIP download is required.

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
python -m tests.compare_auto
```

The two examples generate a synthetic U-shaped bridge and print JSON. To supply aligned arrays and configuration:

```bash
python examples/example.py depth.npy difference.npy examples/config.example.json
python examples/example_auto.py depth.npy difference.npy examples/config.auto.json
```

NPY loading disables pickle. Adjust units, masks, ROIs and thresholds for your own inputs. Execution makes no network requests or robot commands.

## Inputs and outputs

- Two matching depth arrays: current depth and signed background difference. `background-current > 0` means closer to the camera than the background.
- Manual mode: caller-provided hand search ROIs. Automatic candidate mode: a coarse upper search band, with optional arm exclusion. Neither mode recognizes hands semantically.
- Independent outputs: `bridge=yes/no/unknown`, `large_sag=yes/no/unknown`, and `drape=left/right/both/neither/unknown`, plus features, masks and reasons.
- Intrinsics and world/robot calibration are not required for these image-relative estimates. Metric depth units are still required for depth thresholds.

See [API and tuning](API.md) and [bridge-first candidates and comparison](AUTO.md).

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

There are 19 passing synthetic regression tests, including assertions that preserve a known semantic failure. They are not 19 independent real observations. In 18 development scenes, automatic candidates produced 9 matching bridge labels, 1 false bridge and 8 unknown results. The manual-ROI baseline produced 15 matches, 1 false bridge and 2 unknown results, with more caller information.

An arm shaped exactly like a bridge can produce the same depth evidence as a deformable object. An exclusion mask or other semantic input is required in that case. Unknown results include some true bridges. No matched real empty-background/current-depth pairs have been evaluated; real-image accuracy is unvalidated.

## License

Original project code is licensed under [MIT](LICENSE). NumPy is an external dependency under its own BSD-3-Clause license; it is not bundled here. Build tools keep their own licenses. No research images, datasets, third-party source copies or credentials are distributed in this repository. This inventory is not a legal guarantee concerning every possible downstream use.
