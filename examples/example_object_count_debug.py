"""Save group-aware debug PNGs from synthetic scenes, without sensor/robot I/O."""
import argparse
import json
from pathlib import Path
import numpy as np
from depth_image_lens import OverheadObjectCounter


def make_examples():
    """Two touching colors, disconnected matching pieces, and unresolved evidence."""
    shape = (180, 260)
    color = np.full((*shape, 3), 150, np.uint8)
    depth = np.full(shape, 1000, np.uint16)
    empty_depth = depth.copy()
    foreground = np.zeros(shape, bool)
    # A red object with two visible fragments, separated by the blue object.
    foreground[45:140, 25:235] = True
    color[45:140, 25:235] = (210, 35, 35)
    foreground[20:160, 100:155] = True
    color[20:160, 100:155] = (35, 35, 210)
    depth[foreground] = 980
    counter = OverheadObjectCounter()
    evidence = dict(depth=depth, background_depth=empty_depth,
                    depth_scale_m=.001, registered_to_depth=True)
    yield 'groups', color, counter.detect_mask(color, foreground, color_order='RGB', **evidence)
    # Keep color hypotheses even though a bare mask cannot certify a count.
    yield 'mask-only', color, counter.detect_mask(color, foreground, color_order='RGB')
    uncertain = np.zeros(shape, bool)
    uncertain[65:90, 35:65] = True
    excluded = np.zeros(shape, bool)
    excluded[:15, 195:] = True
    with_residue = foreground.copy()
    with_residue[150:170, 20:60] = True
    dark = color.copy()
    dark[150:170, 20:60] = (5, 5, 5)  # Rejected: too dark for color identity.
    yield 'uncertain', dark, counter.detect_mask(
        dark, with_residue, color_order='RGB', uncertain_mask=uncertain,
        exclusion_mask=excluded, diagnostics=True, **evidence)
    yield 'empty', np.full_like(color, 150), counter.detect_mask(
        np.full_like(color, 150), np.zeros(shape, bool), color_order='RGB',
        depth=empty_depth, background_depth=empty_depth, depth_scale_m=.001,
        registered_to_depth=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Existing output directory; files are not overwritten by default.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not args.output_dir.is_dir():
        parser.error('output directory must already exist')
    summaries = {}
    for name, color, result in make_examples():
        result.save_debug(args.output_dir/f'object-count-{name}.png', color,
                          color_order='RGB', panel=False, overwrite=args.overwrite)
        summaries[name] = dict(count=result.count, candidate_count=result.candidate_count,
                               state=result.state)
    print(json.dumps(summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
