"""Image-space hanging shape measurements, not a grasp/cloth-state classifier."""
from dataclasses import dataclass, field
import numpy as np
from .geometry import bool_mask
from .mask_cleanup import _components8


def _finite_number(value, name, minimum=None):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(name + ' must be a finite number')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + ' must be a finite number') from None
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(name + ' is outside its finite range')
    return value


def _quantile_key(q):
    percent = float(q * 100)
    return 'p' + (str(int(percent)) if percent.is_integer() else repr(percent))


def _quantiles(values, qs):
    values = values[np.isfinite(values)]
    return {_quantile_key(q): (float(np.quantile(values, q)) if values.size else None)
            for q in qs}


def _bottom(mask):
    valid = mask.any(axis=0)
    y = mask.shape[0] - 1 - np.argmax(mask[::-1], axis=0)
    return np.where(valid, y, np.nan)


def _run_supported(mask, minimum):
    # Every pixel of each sufficiently long vertical run is retained. No holes
    # are filled and no new foreground observations are introduced.
    out = np.zeros_like(mask)
    for x in np.flatnonzero(mask.any(axis=0)):
        edges = np.flatnonzero(np.diff(np.r_[False, mask[:, x], False]))
        for start, end in edges.reshape(-1, 2):
            if end - start >= minimum:
                out[start:end, x] = True
    return out


@dataclass(frozen=True)
class HangingProfileResult:
    profiles: dict = field(repr=False)
    metrics: dict
    parameters: dict
    warnings: list
    diagnostics: dict = field(default_factory=dict, repr=False)
    scope: str = 'visible_image_space_hanging_shape_not_grasp_or_fold_classification'
    validation_status: str = 'not_validated_on_real_images'

    def to_dict(self, *, include_profiles=False):
        """JSON-safe summary; optional profiles encode non-finite values as null."""
        result = dict(metrics=self.metrics.copy(), parameters=self.parameters.copy(),
                      warnings=list(self.warnings), scope=self.scope,
                      validation_status=self.validation_status)
        if include_profiles:
            result['profiles'] = {k: [None if isinstance(x, float) and not np.isfinite(x) else x
                                       for x in v.tolist()] for k, v in self.profiles.items()}
        return result

    def render_debug(self, color=None):
        """RGB image: observed mask, raw/supported/smoothed bottom and row widths.

        Requires analyze_mask(..., diagnostics=True) and optional OpenCV.
        """
        if not self.diagnostics:
            raise ValueError('render_debug requires diagnostics=True')
        try:
            import cv2
        except ImportError as exc:
            raise ImportError('install depth-image-lens[debug] for debug images') from exc
        mask = self.diagnostics['retained_observed_mask']
        h, w = mask.shape
        if color is None:
            canvas = np.zeros((h, w, 3), np.uint8)
        else:
            color = np.asarray(color)
            if color.shape != (h, w, 3) or color.dtype != np.uint8:
                raise ValueError('color must be aligned uint8 HxWx3 RGB')
            canvas = color.copy()
        canvas[mask] = ((canvas[mask].astype(float) + (40, 150, 60)) / 2).astype(np.uint8)
        canvas[self.diagnostics['uncertain_foreground_mask']] = (160, 70, 180)
        canvas[self.diagnostics['removed_small_component_mask']] = (120, 120, 120)
        x0, x1 = self.parameters['expected_x_range']
        yref = self.parameters['reference_y']
        if 0 <= yref < h:
            cv2.line(canvas, (x0, int(round(yref))), (x1 - 1, int(round(yref))), (255, 255, 255), 1)
        for key, rgb in [('raw_bottom_y_px', (255, 150, 30)),
                         ('bottom_y_px', (40, 220, 255)),
                         ('smoothed_bottom_y_px', (255, 255, 40))]:
            for x in np.flatnonzero(np.isfinite(self.profiles[key])):
                y = int(round(self.profiles[key][x]))
                if 0 <= y < h:
                    canvas[y, x] = rgb
        for x in np.flatnonzero(self.profiles['bottom_censored']):
            y = int(self.profiles['bottom_y_px'][x])
            cv2.circle(canvas, (int(x), y), 2, (255, 70, 200), 1)
        # Render at native scale, with a separate row-width chart and legend.
        out = np.full((h + 100, max(720, w + 160), 3), 25, np.uint8)
        out[:h, :w] = canvas
        widths = self.profiles['row_width_px']
        for y in range(h):
            length = int(140 * widths[y] / max(1, x1 - x0))
            out[y, w + 5:w + 5 + length] = (40, 170, 80)
        lines = ['Orange: raw  Cyan: supported  Yellow: median-smoothed',
                 'Pink circle: lower bound / incomplete observation',
                 'Purple: inferred/uncertain  Gray: removed small component',
                 'White: reference y   Right chart: observed row width']
        for i, line in enumerate(lines):
            cv2.putText(out, line, (5, h + 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        .38, (240, 240, 240), 1, cv2.LINE_AA)
        return out


class HangingProfileAnalyzer:
    """Measure a caller-selected hanging region, image y increasing downward.

    Analyze each label separately, or explicitly union fragments belonging to the
    same object. This class does not infer ownership or locate grippers. Raw and
    supported profiles describe visible observations, never hidden cloth length.
    """

    def __init__(self, *, min_component_area=10, min_vertical_run_px=3,
                 smoothing_window_px=5, quantiles=(.1, .25, .5, .75, .9),
                 reach_drop_px=50., upper_band_height_px=25.):
        for name, value in dict(min_component_area=min_component_area,
                                min_vertical_run_px=min_vertical_run_px,
                                smoothing_window_px=smoothing_window_px).items():
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if smoothing_window_px % 2 != 1:
            raise ValueError('smoothing_window_px must be odd')
        try:
            qs = tuple(_finite_number(q, 'quantiles') for q in quantiles)
        except TypeError:
            raise ValueError('quantiles must be a nonempty sequence') from None
        if not qs or len(set(qs)) != len(qs) or any(q < 0 or q > 1 for q in qs):
            raise ValueError('quantiles must be distinct values in [0, 1]')
        if len({_quantile_key(q) for q in qs}) != len(qs):
            raise ValueError('quantile percentage keys must be distinguishable')
        self.parameters = dict(min_component_area=min_component_area,
                               min_vertical_run_px=min_vertical_run_px,
                               smoothing_window_px=smoothing_window_px,
                               quantiles=list(qs),
                               reach_drop_px=_finite_number(reach_drop_px, 'reach_drop_px', 0),
                               upper_band_height_px=_finite_number(upper_band_height_px, 'upper_band_height_px', 0))

    def analyze_mask(self, foreground_mask, *, reference_y, expected_x_range=None,
                     observed_mask=None, uncertain_mask=None, mask_complete=False,
                     diagnostics=False):
        """Measure below reference_y, optionally within [x0, x1).

        observed_mask marks observed pixels (including observable background),
        uncertain_mask marks inferred repairs/uncertain membership. False values
        of the object mask do NOT imply absence unless mask_complete=True.
        Missing/unsupported columns are NaN, never fabricated zeros. Censored
        columns have a visible lower bound, not a known physical bottom.
        A lower-bound profile does not quantify undetected segmentation errors.
        """
        source = np.asarray(foreground_mask)
        if source.dtype != bool or source.ndim != 2 or min(source.shape) < 1:
            raise ValueError('foreground_mask must be a nonempty boolean HxW array')
        if type(mask_complete) is not bool or type(diagnostics) is not bool:
            raise ValueError('mask_complete and diagnostics must be bool')
        h, w = source.shape
        ref = _finite_number(reference_y, 'reference_y')
        if ref >= h:
            raise ValueError('reference_y must leave a visible below-reference region')
        if expected_x_range is None:
            x0, x1 = 0, w
        else:
            if not isinstance(expected_x_range, (tuple, list)) or len(expected_x_range) != 2:
                raise ValueError('expected_x_range must be (x0, x1)')
            x0, x1 = expected_x_range
            if any(type(x) is not int for x in (x0, x1)) or not 0 <= x0 < x1 <= w:
                raise ValueError('expected_x_range must satisfy 0 <= x0 < x1 <= width')
        observed = np.ones(source.shape, bool) if observed_mask is None else bool_mask(observed_mask, source.shape, 'observed_mask')
        uncertain = np.zeros(source.shape, bool) if uncertain_mask is None else bool_mask(uncertain_mask, source.shape, 'uncertain_mask')
        domain = np.zeros(source.shape, bool)
        start_y = max(0, int(np.ceil(ref)))
        domain[start_y:, x0:x1] = True
        visible = source & observed & ~uncertain & domain
        labels, sizes = _components8(visible)
        keep = sizes >= self.parameters['min_component_area']
        keep[0] = False
        retained = keep[labels]
        supported = _run_supported(retained, self.parameters['min_vertical_run_px'])
        raw_bottom = _bottom(visible)
        bottom = _bottom(supported)
        valid = np.isfinite(bottom)
        smooth = np.full(w, np.nan)
        radius = self.parameters['smoothing_window_px'] // 2
        # Smooth only across uninterrupted supported columns; do not bridge gaps.
        for x in np.flatnonzero(valid):
            left, right = x, x + 1
            while left > max(x0, x - radius) and valid[left - 1]:
                left -= 1
            while right < min(x1, x + radius + 1) and valid[right]:
                right += 1
            smooth[x] = np.median(bottom[left:right])
        blocked = (~observed | uncertain) & domain
        censored = np.zeros(w, bool)
        for x in np.flatnonzero(valid):
            y = int(bottom[x])
            censored[x] = (not mask_complete or y == h - 1 or blocked[y + 1:, x].any()
                           or raw_bottom[x] > y)
        exact = valid & ~censored
        drop = bottom - ref
        exact_drop = np.where(exact, drop, np.nan)
        row_width = retained.sum(axis=1)
        reach = valid & (drop >= self.parameters['reach_drop_px'])
        coverage = np.full(w, np.nan)
        coverage[x0:x1] = (~blocked[start_y:, x0:x1]).mean(axis=0)
        area = int(retained.sum())
        upper = np.arange(h)[:, None] < ref + self.parameters['upper_band_height_px']
        upper_area = int((retained & upper).sum())
        finite_drop = drop[valid]
        q = self.parameters['quantiles']
        raw_components = int(len(sizes) - 1)
        kept_components = int(keep.sum())
        edge_flags = dict(left=bool(retained[:, x0].any()), right=bool(retained[:, x1 - 1].any()),
                          bottom=bool(retained[-1].any()))
        warnings = []
        if not mask_complete:
            warnings.append('mask_completeness_not_attested')
        if observed_mask is None:
            warnings.append('full_pixel_observation_assumed')
        if not valid.any():
            warnings.append('no_supported_foreground_columns')
        if blocked.any():
            warnings.append('missing_or_uncertain_pixels_in_measurement_domain')
        if kept_components > 1:
            warnings.append('multiple_fragments_not_automatically_associated')
        if any(edge_flags.values()):
            warnings.append('foreground_touches_measurement_boundary')
        if censored.any():
            warnings.append('bottom_profile_contains_lower_bounds')
        if np.any(visible & ~supported):
            warnings.append('small_or_short_support_removed_for_profile')
        metrics = dict(
            reference_width_px=x1 - x0, observed_area_px=area,
            raw_component_count=raw_components, retained_component_count=kept_components,
            removed_small_component_pixels=int((visible & ~retained).sum()),
            unsupported_profile_pixels=int((retained & ~supported).sum()),
            inferred_foreground_pixels=int((source & uncertain & domain).sum()),
            supported_columns=int(valid.sum()), exact_columns=int(exact.sum()),
            censored_columns=int(censored.sum()),
            supported_width_fraction=float(valid.sum() / (x1 - x0)),
            observation_fraction=float((~blocked & domain).sum() / domain.sum()),
            reaching_width_fraction=float(reach.sum() / (x1 - x0)),
            reaching_observed_columns_fraction=float(reach.sum() / valid.sum()) if valid.any() else None,
            upper_area_fraction=float(upper_area / area) if area else None,
            upper_area_px=upper_area,
            drop_quantiles_px=_quantiles(drop, q),
            exact_drop_quantiles_px=_quantiles(exact_drop, q),
            drop_mean_px=float(finite_drop.mean()) if finite_drop.size else None,
            drop_iqr_px=float(np.quantile(finite_drop, .75) - np.quantile(finite_drop, .25)) if finite_drop.size else None,
            drop_mad_px=float(np.median(np.abs(finite_drop - np.median(finite_drop)))) if finite_drop.size else None,
            boundary_touches=edge_flags,
            complete_bottom_distribution=bool(mask_complete and valid.any() and not censored.any()
                                              and not blocked.any()
                                              and not (np.isfinite(raw_bottom) & ~valid).any()),
            classification='not_performed')
        profiles = dict(x_px=np.arange(w), raw_bottom_y_px=raw_bottom, bottom_y_px=bottom,
                        smoothed_bottom_y_px=smooth, drop_px=drop,
                        smoothed_drop_px=smooth - ref, exact_drop_px=exact_drop,
                        bottom_censored=censored, row_y_px=np.arange(h), row_width_px=row_width,
                        row_width_fraction=row_width / (x1 - x0),
                        column_observed_fraction=coverage, reaching_columns=reach)
        debug = dict(input_mask=source.copy(), measurement_domain=domain,
                     observed_mask=observed.copy(), retained_observed_mask=retained,
                     supported_profile_mask=supported, removed_small_component_mask=visible & ~retained,
                     uncertain_foreground_mask=source & uncertain & domain,
                     missing_observation_mask=~observed & domain) if diagnostics else {}
        return HangingProfileResult(profiles, metrics, {**self.parameters, 'reference_y': ref,
                                    'expected_x_range': [x0, x1], 'mask_complete': mask_complete}, warnings, debug)
