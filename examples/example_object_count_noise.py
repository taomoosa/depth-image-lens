"""Small arm/noise counterexamples, not a field benchmark; no files or robot I/O."""
import json
import numpy as np
from depth_image_lens import OverheadObjectCounter


RED = (210, 35, 35)
RESIDUE = (200, 140, 100)


def scene():
    shape = (160, 220)
    color = np.full((*shape, 3), 150, np.uint8)
    depth = np.full(shape, 1000, np.uint16)
    return color, depth, depth.copy(), np.zeros(shape, bool), {}


def rectangle(arrays, name, box, color=RED, separation_mm=20):
    rgb, depth, _, foreground, regions = arrays
    region = np.zeros(depth.shape, bool)
    region[box] = True
    rgb[region] = color
    depth[region] = 1000 - separation_mm
    foreground[region] = True
    regions[name] = region
    return region


def run(arrays, *, arm_mask=None, **parameters):
    color, depth, empty, foreground, regions = arrays
    result = OverheadObjectCounter(**parameters).detect_mask(
        color, foreground, color_order='RGB', depth=depth, background_depth=empty,
        depth_scale_m=.001, registered_to_depth=True, arm_mask=arm_mask)
    return dict(
        count=result.count, candidate_count=result.candidate_count, reasons=result.reasons,
        parameters=parameters,
        pixels={name: dict(input=int(region.sum()), grouped=int((region & result.object_mask).sum()),
                           excluded=int((region & result.excluded_mask).sum()))
                for name, region in regions.items()},
    )


def run_examples():
    examples = {}
    arrays = scene()
    rectangle(arrays, 'target', np.s_[30:60, 20:50])
    rectangle(arrays, 'arm', np.s_[40:120, 120:160], RESIDUE, 300)
    arm = np.zeros(arrays[1].shape, bool)
    arm[43:117, 123:157] = True  # Under-covers every edge by three pixels.
    for guard in (2, 3):
        examples[f'arm_edge_guard_{guard}'] = run(arrays, arm_mask=arm, exclusion_guard_px=guard)

    arrays = scene()
    arm = rectangle(arrays, 'arm', np.s_[40:120, 80:110], RESIDUE, 300)
    rectangle(arrays, 'target', np.s_[60:100, 110:112])  # Real two-pixel-wide strip.
    for guard in (0, 2):
        examples[f'adjacent_strip_guard_{guard}'] = run(arrays, arm_mask=arm, exclusion_guard_px=guard)

    arrays = scene()
    rectangle(arrays, 'target', np.s_[10:40, 10:40])
    residual = np.zeros(arrays[1].shape, bool)
    for i in range(8):
        residual[60:63, 10+12*i:13+12*i] = True
    arrays[0][residual] = RESIDUE
    arrays[1][residual] = 980
    arrays[3][residual] = True
    arrays[4]['residue'] = residual
    for minimum in (8, 10):
        examples[f'eight_fragments_min_{minimum}'] = run(arrays, min_fragment_area=minimum)

    arrays = scene()
    rectangle(arrays, 'target', np.s_[20:50, 20:50], separation_mm=200)
    rectangle(arrays, 'arm', np.s_[70:110, 120:160], RESIDUE, 20)
    examples['upper_band_wrong_identity'] = run(arrays, max_reference_separation_m=.15,
                                               exclusion_guard_px=0)

    arrays = scene()
    rectangle(arrays, 'target', np.s_[10:30, 10:30])
    rectangle(arrays, 'residue', np.s_[40:60, 60:80], RED)
    examples['same_color_contamination'] = run(arrays)
    return examples


if __name__ == '__main__':
    print(json.dumps(run_examples(), indent=2, allow_nan=False))
