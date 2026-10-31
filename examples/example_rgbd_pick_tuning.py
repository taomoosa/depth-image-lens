"""Run RGB-D mask/pick tuning with recorded settings, masks and pixel evidence.

No input file: generated scene. With --input: an NPZ of registered current/empty
arrays; explicit --depth-scale-m and --color-order are required. Outputs go only
to a new user-selected directory. This example does not operate a camera.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector, OverheadPickCandidateDetector


MASK_KEYS = ('color_valid_mask', 'background_color_valid_mask', 'depth_valid_mask',
             'background_depth_valid_mask', 'work_roi', 'background_check_mask')


def demo():
    color = np.full((100, 140, 3), 140, np.uint8)
    depth = np.full((100, 140), 1000, np.uint16)
    data = dict(background_color=color.copy(), background_depth=depth.copy())
    obj = np.zeros(depth.shape, bool)
    obj[10:90, 10:105] = True; obj[30:58, 55:82] = False
    color[obj] = (35, 170, 90); depth[obj] = 980
    depth[30:60, 23:47] = 940
    depth[75, 85] = 300
    depth[65:70, 25:30] = 0
    color[38:50, 63:75] = (40, 80, 210); depth[38:50, 63:75] = 500
    # A known unchanged region permits optional color-offset compensation.
    check = np.zeros(depth.shape, bool); check[:5, :30] = True
    data.update(color=color, depth=depth, background_check_mask=check)
    return data


def finite_scalar(value):
    return value.item() if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='NPZ: color, depth, background_color, background_depth')
    parser.add_argument('--depth-scale-m', type=float, help='Metres/count in BOTH depth arrays; demo uses .001')
    parser.add_argument('--color-order', choices=('RGB', 'BGR'), help='Order of BOTH colors; demo uses RGB')
    parser.add_argument('--config', type=Path, help='JSON object with segmenter and/or pick constructor overrides')
    parser.add_argument('--contour-id', type=int, help='ID from outline.json; default largest polygon area')
    parser.add_argument('--inspect-x', type=int, help='Optional pixel x to inspect even when no candidate exists')
    parser.add_argument('--inspect-y', type=int, help='Optional pixel y to inspect')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; parent must exist')
    args = parser.parse_args()
    if (args.inspect_x is None) != (args.inspect_y is None):
        parser.error('supply both --inspect-x and --inspect-y')
    if args.input and (args.depth_scale_m is None or args.color_order is None):
        parser.error('--input requires explicit --depth-scale-m and --color-order')
    if args.output_dir.exists() or not args.output_dir.parent.is_dir():
        parser.error('--output-dir must be new and its parent must exist')
    config = {}
    if args.config:
        def reject_constant(value): raise ValueError('nonfinite JSON constant: ' + value)
        config = json.loads(args.config.read_text(), parse_constant=reject_constant)
        if not isinstance(config, dict) or set(config) - {'segmenter', 'pick'}:
            parser.error('config must contain only segmenter and/or pick objects')
        if any(not isinstance(value, dict) for value in config.values()):
            parser.error('segmenter and pick settings must be JSON objects')
    if args.input:
        with np.load(args.input, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        for key in ('color', 'depth', 'background_color', 'background_depth'):
            if key not in data: parser.error('missing NPZ key: ' + key)
        scale, order = args.depth_scale_m, args.color_order
    else:
        if args.depth_scale_m is not None or args.color_order is not None:
            parser.error('generated demo uses fixed RGB and .001 m/count; unit/order overrides require --input')
        data = demo(); scale, order = .001, 'RGB'
    segmenter = RGBDBackgroundSegmenter(**config.get('segmenter', {}))
    picker = OverheadPickCandidateDetector(**config.get('pick', {}))
    seg = segmenter.detect(data['color'], data['depth'],
        background_color=data['background_color'], background_depth=data['background_depth'],
        color_order=order, depth_scale_m=scale, registered_to_depth=True,
        invalid_values=data.get('invalid_values', ()), diagnostics=True,
        **{key: data[key] for key in MASK_KEYS if key in data})
    outline = OuterContourDetector().detect(seg.object_mask, observed_mask=seg.observed_mask)
    pick = picker.detect(data['depth'], outline, contour_id=args.contour_id,
        background_depth=data['background_depth'], depth_scale_m=scale, registered_to_mask=True,
        depth_valid_mask=seg.depth_valid_mask, segmentation_uncertain_mask=seg.uncertain_mask,
        invalid_values=data.get('invalid_values', ()), diagnostics=True)
    point = [args.inspect_x, args.inspect_y] if args.inspect_x is not None else pick.point_xy
    evidence = None
    if point is not None:
        x, y = point
        if not (0 <= y < seg.object_mask.shape[0] and 0 <= x < seg.object_mask.shape[1]):
            parser.error('inspection pixel is outside the image')
        evidence = dict(point_xy=point, segmenter={}, pick={})
        for name, result in [('segmenter', seg), ('pick', pick)]:
            for key, value in {**vars(result), **result.diagnostics}.items():
                if isinstance(value, np.ndarray) and value.shape == seg.object_mask.shape:
                    evidence[name][key] = finite_scalar(value[y, x])
    # Validate and compute first; only explicit output creates files.
    args.output_dir.mkdir()
    def write_json(name, value):
        (args.output_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    write_json('parameters.json', dict(segmenter=seg.parameters, pick=pick.parameters))
    write_json('segmentation.json', seg.to_dict()); write_json('outline.json', outline.to_dict())
    write_json('pick.json', pick.to_dict()); write_json('pixel.json', evidence)
    summary = dict(input_kind='provided_arrays' if args.input else 'generated_scene',
                   depth_scale_m=scale, color_order=order, contour_id=pick.contour_id,
                   point_xy=pick.point_xy, pick_state=pick.state,
                   segmentation_metrics=seg.metrics, pick_metrics=pick.metrics)
    write_json('summary.json', summary)
    arrays = {}
    for name, result in [('seg', seg), ('pick', pick)]:
        arrays.update({name+'_'+key: value for key, value in vars(result).items() if isinstance(value, np.ndarray)})
        arrays.update({name+'_diag_'+key: value for key, value in result.diagnostics.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(args.output_dir/'arrays.npz', **arrays)
    seg.save_debug(args.output_dir/'segmentation.png', data['color'], color_order=order)
    pick.save_debug(args.output_dir/'pick.png', data['color'], color_order=order)
    outline.save_debug(args.output_dir/'outline.png', data['color'], color_order=order)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
