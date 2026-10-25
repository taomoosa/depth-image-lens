"""Local NPY/NPZ runner; inputs are data, never executable configuration."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import BridgeDrapeDetector
from .depth import (background_difference, _numeric_image, _invalid_values,
                    depth_pair_16uc1_to_meters, positive_difference_16uc1)
from .diagnostics import save_debug

MASKS = ('background_valid', 'observed_mask', 'arm_mask', 'work_roi')


def _load_npy(path):
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError('separate inputs must be NPY arrays')
    if array.dtype.hasobject:
        raise ValueError('object arrays are not accepted')
    return array


def _inputs(args):
    if args.input:
        if args.depth or args.background or args.difference:
            raise ValueError('use either --input NPZ or separate NPY paths')
        archive = np.load(args.input, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError('--input requires an NPZ archive')
        with archive:
            keys = set(archive.files)
            if len(keys) != len(archive.files) or not keys <= {'depth', 'background', 'difference', *MASKS}:
                raise ValueError('NPZ contains duplicate or unsupported keys')
            if 'depth' not in keys or (('background' in keys) == ('difference' in keys)):
                raise ValueError('NPZ requires depth and exactly one of background/difference')
            arrays = {key: archive[key] for key in keys}
            if any(a.dtype.hasobject for a in arrays.values()):
                raise ValueError('object arrays are not accepted')
            return arrays
    if not args.depth or bool(args.background) == bool(args.difference):
        raise ValueError('provide --depth and exactly one of --background/--difference')
    other = 'background' if args.background else 'difference'
    return {'depth': _load_npy(args.depth), other: _load_npy(getattr(args, other))}


def _reject_constant(value):
    raise ValueError('configuration must be strict JSON; nonfinite constants are not allowed')


def run(arrays, config, config_directory, *, diagnostics=True):
    """Return a result without storing source paths or user metadata in it."""
    if not isinstance(config, dict):
        raise ValueError('configuration must be a JSON object')
    options = dict(config)
    mode = options.pop('mode', 'auto')
    if mode not in ('manual', 'auto'):
        raise ValueError('mode must be manual or auto')
    parameters = options.pop('detector', {})
    if not isinstance(parameters, dict):
        raise ValueError('detector must be a JSON object')
    options.pop('diagnostics', None)  # Caller controls optional recording; CLI enables it.
    sentinels = _invalid_values(options.pop('invalid_values', ()))
    encoding = options.pop('encoding', None)
    encoded_valid = None
    if encoding is not None:
        if encoding != '16UC1': raise ValueError('supported encoding is 16UC1')
        if set(options) & {'depth_unit', 'background_unit', 'difference_unit', 'difference_sign'}:
            raise ValueError('16UC1 encoding uses one depth_scale_m; do not mix unit/sign settings')
        if options.get('difference_mode', 'positive_foreground') != 'positive_foreground':
            raise ValueError('16UC1 convenience requires positive_foreground mode')
        arrays = dict(arrays)
        if 'background' in arrays:
            delta, encoded_valid = positive_difference_16uc1(arrays['depth'], arrays.pop('background'), invalid_values=sentinels)
        else:
            delta = arrays['difference']
            encoded_valid = ~np.isin(arrays['depth'], sentinels)
        arrays['depth'], arrays['difference'] = depth_pair_16uc1_to_meters(
            arrays['depth'], delta, depth_scale_m=options.pop('depth_scale_m', 0.001))
        options.update(depth_unit='m', difference_unit='m', difference_sign=None, difference_mode='positive_foreground')
        sentinels = np.asarray(())  # Sentinel values above were raw counts, not metres.
    elif 'depth_scale_m' in options:
        raise ValueError('depth_scale_m requires explicit encoding=16UC1')
    depth = _numeric_image(arrays['depth'], 'depth')
    for key in MASKS:
        if key in arrays and key in options:
            raise ValueError(f'{key} must be supplied in either NPZ or config, not both')
        if key in arrays:
            options[key] = arrays[key]
        elif key in options:
            if not isinstance(options[key], str):
                raise ValueError(f'{key} in config must be an NPY path')
            options[key] = _load_npy(Path(config_directory) / options[key])
        if key in options and (options[key].dtype != bool or options[key].shape != depth.shape):
            raise ValueError(f'{key} must be a matching bool mask')
    if encoded_valid is not None:
        options['background_valid'] = encoded_valid & options.get('background_valid', True)
    if 'depth_unit' not in options:
        raise ValueError('depth_unit is required; units are never inferred')
    if 'background' in arrays:
        if options.get('difference_mode', 'signed') != 'signed':
            raise ValueError('background input produces signed differences; use precomputed input for positive_foreground')
        if 'background_unit' not in options:
            raise ValueError('background_unit is required for background input')
        unit = options['depth_unit']
        if options.get('difference_unit', unit) != unit or options.get('difference_sign', 'background-current') != 'background-current':
            raise ValueError('computed difference uses matching units and background-current sign')
        difference, valid = background_difference(depth, arrays['background'], depth_unit=unit,
            background_unit=options.pop('background_unit'), invalid_values=sentinels)
        options['background_valid'] = valid & options.get('background_valid', True)
        options.update(difference_unit=unit, difference_sign='background-current')
    else:
        if 'background_unit' in options:
            raise ValueError('background_unit is only valid for background input')
        if 'difference_unit' not in options:
            raise ValueError('precomputed difference requires difference_unit')
        if options.get('difference_mode', 'signed') == 'positive_foreground':
            options.setdefault('difference_sign', None)
        elif 'difference_sign' not in options:
            raise ValueError('signed difference requires difference_sign')
        difference = _numeric_image(arrays['difference'], 'difference')
        # Sentinels describe raw depth, never difference zero (known background).
        options['observed_mask'] = ~np.isin(depth, sentinels) & options.get('observed_mask', True)
    detector = BridgeDrapeDetector(**parameters)
    return getattr(detector, 'detect' if mode == 'manual' else 'detect_auto')(
        depth, difference, diagnostics=diagnostics, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='NPZ: depth plus background or difference; optional bool masks')
    parser.add_argument('--depth', help='Current depth NPY')
    parser.add_argument('--background', help='Empty background depth NPY')
    parser.add_argument('--difference', help='Precomputed difference NPY; signed or positive_foreground mode in config')
    parser.add_argument('--config', required=True, help='JSON units, mode and detector options')
    parser.add_argument('--output', default='debug-output', help='Local diagnostic output directory')
    args = parser.parse_args(argv)
    try:
        path = Path(args.config)
        config = json.loads(path.read_text(), parse_constant=_reject_constant)
        result = run(_inputs(args), config, path.parent)
        payload = json.dumps(result.to_dict(), allow_nan=False)
        save_debug(result, args.output)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    print(payload)


if __name__ == '__main__':
    main()
