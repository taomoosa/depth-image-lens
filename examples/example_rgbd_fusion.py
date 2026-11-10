"""Compare RGB-D fusion gates on generated evidence and known counterexamples."""
import argparse
import json
from pathlib import Path
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector


def scenes():
    """Independent synthetic cases; masks are construction labels, not real GT."""
    for name in ('both_weak', 'weak_depth_same_color', 'isolated_depth',
                 'strong_same_color', 'thin_chroma', 'luma_shadow',
                 'missing_depth_chroma', 'same_color_line', 'correlated_error',
                 'color_misalignment', 'farther_conflict'):
        bc = np.full((40, 60, 3), 140, np.uint8)
        bz = np.full((40, 60), 1000, np.uint16)
        c, z = bc.copy(), bz.copy()
        region = (slice(8, 25), slice(8, 30))
        if name in ('both_weak', 'weak_depth_same_color', 'correlated_error'):
            z[region] = 993
        if name == 'both_weak':
            c[region] = (30, 150, 210)
        elif name == 'isolated_depth':
            z[15, 15] = 900
        elif name == 'strong_same_color':
            z[region] = 970
            z[8:13, 23:30] = 1000  # Concavity.
            z[16:20, 15:19] = 1000  # Hole.
            z[28:35, 43:52] = 970  # Separate component.
        elif name in ('thin_chroma', 'missing_depth_chroma'):
            c[region] = (30, 150, 210)
            z[region] = 999 if name == 'thin_chroma' else 0
        elif name in ('luma_shadow', 'correlated_error'):
            c[region] = 40
        elif name == 'same_color_line':
            z[15, 8:30] = 970
        elif name == 'color_misalignment':
            bc[region] = (30, 150, 210)
            c[8:25, 10:32] = (30, 150, 210)
        elif name == 'farther_conflict':
            c[region] = (30, 150, 210)
            z[region] = 1100
        yield name, c, z, bc, bz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='New directory; parent must exist. Saves arrays and debug PNGs.')
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir()  # Explicit, no overwrite.
    rows = []
    for name, color, depth, empty_color, empty_depth in scenes():
        row = dict(case=name)
        for mode in ('union', 'depth_required', 'evidence'):
            seg = RGBDBackgroundSegmenter(fusion_mode=mode).detect(
                color, depth, background_color=empty_color, background_depth=empty_depth,
                color_order='RGB', registered_to_depth=True, depth_scale_m=.001, diagnostics=True)
            outline = OuterContourDetector().detect(seg.object_mask, observed_mask=seg.outline_observed_mask)
            row[mode] = dict(selected=int(seg.object_mask.sum()),
                suppressed=int(seg.suppressed_mask.sum()), deferred=int(seg.deferred_mask.sum()),
                selected_uncertain=int((seg.object_mask & seg.uncertain_mask).sum()),
                contours=len(outline.contours), holes=len(outline.holes))
            if args.output_dir:
                prefix = args.output_dir / (name + '-' + mode)
                seg.save_debug(prefix.with_suffix('.png'), color)
                prefix.with_suffix('.json').write_text(json.dumps(seg.to_dict(), indent=2, allow_nan=False) + '\n')
                arrays = {key: value for key, value in vars(seg).items() if isinstance(value, np.ndarray)}
                arrays.update({'diag_' + key: value for key, value in seg.diagnostics.items() if isinstance(value, np.ndarray)})
                arrays['outline_observed_mask'] = seg.outline_observed_mask
                np.savez_compressed(prefix.with_suffix('.npz'), **arrays)
        rows.append(row)
    result = dict(scope='synthetic_rule_comparison_not_real_image_accuracy', comparisons=rows)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output_dir:
        (args.output_dir / 'comparison.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
