"""Explicit-unit background subtraction for registered, fixed-camera depth."""
import numpy as np


def _numeric_image(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 2 or array.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a real numeric HxW array, not object/bool/complex')
    return array.astype(np.float64, copy=False)


def _invalid_values(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise ValueError('invalid_values must be a finite numeric list')
    return values


def background_difference(depth, background, *, depth_unit, background_unit, invalid_values=()):
    """Return (background-current, valid_mask) in the declared input unit.

    Both arrays must be registered HxW depth in the same explicit m/mm unit.
    Subtraction is float64 even for uint16 inputs. Either nonpositive/nonfinite
    depth, or a declared sentinel in either input, makes the difference NaN and
    the mask false. No registration, unit inference or hole filling is performed.
    """
    if depth_unit not in ('m', 'mm') or background_unit != depth_unit:
        raise ValueError('declare matching m/mm depth_unit and background_unit')
    current = _numeric_image(depth, 'depth')
    empty = _numeric_image(background, 'background')
    if current.shape != empty.shape:
        raise ValueError('depth and background must have the same shape')
    sentinels = _invalid_values(invalid_values)
    valid = np.isfinite(current) & np.isfinite(empty) & (current > 0) & (empty > 0)
    valid &= ~np.isin(current, sentinels) & ~np.isin(empty, sentinels)
    with np.errstate(over='ignore', invalid='ignore'):
        difference = empty - current
    valid &= np.isfinite(difference)
    difference[~valid] = np.nan
    return difference, valid


def _uint16_image(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 2 or array.dtype.kind != 'u' or array.dtype.itemsize != 2:
        raise ValueError(f'{name} must be a single-channel uint16 HxW array (16UC1)')
    return array


def depth_pair_16uc1_to_meters(depth, difference, *, depth_scale_m=0.001):
    """Decode two 16UC1 arrays using one explicit metres-per-count scale.

    The 1 mm/count default belongs only to this convenience entrypoint, never
    to dtype-based inference in the general m/mm detector API. Difference zero
    is preserved as observed exclusion; current-depth zero remains invalid.
    """
    current = _uint16_image(depth, 'depth')
    delta = _uint16_image(difference, 'difference')
    if current.shape != delta.shape:
        raise ValueError('16UC1 depth and difference must have matching shapes')
    return (_depth_16uc1_to_meters(current, depth_scale_m=depth_scale_m),
            _depth_16uc1_to_meters(delta, depth_scale_m=depth_scale_m))


def _depth_16uc1_to_meters(depth, *, depth_scale_m=0.001):
    """Shared count conversion, with no in-place changes to caller arrays."""
    current = _uint16_image(depth, 'depth')
    _validate_depth_scale(depth_scale_m)
    with np.errstate(over='ignore', invalid='ignore'):
        current_m = current.astype(np.float64)
        current_m *= depth_scale_m
    if not np.isfinite(current_m).all():
        raise ValueError('depth_scale_m overflows metric depth')
    return current_m


def positive_difference_16uc1(depth, background, *, invalid_values=()):
    """Return clipped uint16 background-current counts and a separate valid mask.

    Int32 subtraction prevents unsigned wraparound. Either zero/sentinel input
    is invalid and outputs zero, so callers MUST retain the returned valid mask.
    The clipped image alone cannot distinguish missing measurements from valid
    background/excluded pixels. Both inputs must use the same count scale.
    """
    current = _uint16_image(depth, 'depth')
    empty = _uint16_image(background, 'background')
    if current.shape != empty.shape:
        raise ValueError('16UC1 depth and background must have matching shapes')
    sentinels = _invalid_values(invalid_values)
    valid = (current > 0) & (empty > 0) & ~np.isin(current, sentinels) & ~np.isin(empty, sentinels)
    delta = np.maximum(empty.astype(np.int32) - current.astype(np.int32), 0)
    delta[~valid] = 0
    return delta.astype(np.uint16), valid


def _validate_depth_scale(depth_scale_m):
    if not np.isscalar(depth_scale_m) or np.asarray(depth_scale_m).dtype.kind not in 'iuf' or not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise ValueError('depth_scale_m must be a positive finite scalar')
