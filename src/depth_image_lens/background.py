"""Fixed-view RGB-D empty-scene subtraction, with explicit evidence provenance."""
from dataclasses import dataclass, field
import numpy as np
from .boundary import _background_selection, _components
from .debug import DebugResultMixin, _opencv
from .depth import _numeric_image, _validate_depth_scale, _invalid_values
from .geometry import bool_mask


@dataclass(frozen=True)
class RGBDBackgroundResult(DebugResultMixin):
    object_mask: np.ndarray = field(repr=False)
    uncertain_mask: np.ndarray = field(repr=False)
    observed_mask: np.ndarray = field(repr=False)
    color_foreground_mask: np.ndarray = field(repr=False)
    depth_foreground_mask: np.ndarray = field(repr=False)
    depth_valid_mask: np.ndarray = field(repr=False)
    conflict_mask: np.ndarray = field(repr=False)
    parameters: dict
    stages: dict
    metrics: dict
    diagnostics: dict = field(default_factory=dict, repr=False)
    scope: str = 'registered_change_candidates_not_semantic_objects_or_verified_material_edges'

    def to_dict(self):
        return {k: v for k, v in vars(self).items()
                if not isinstance(v, np.ndarray) and k != 'diagnostics'}

    def render_debug(self, color=None, *, color_order='RGB', **options):
        """Common mask panels plus fixed-scale evidence maps when recorded."""
        base = super().render_debug(color, color_order=color_order, **options)
        if not self.diagnostics:
            return base
        cv = _opencv()
        width = max(280, base.shape[1] // 3)
        maps = [('background_combined_score', 'Lab score 0..4; threshold 1', 4., False),
                ('depth_difference_m', 'Empty-current Z; +/-2T',
                 self.parameters['depth_threshold_m'], True)]
        tiles = []
        for key, title, scale, signed in maps:
            values = self.diagnostics[key]
            valid = np.isfinite(values)
            gray = np.zeros(values.shape, np.uint8)
            # Divide by T before applying the display factor to avoid 2*T
            # overflow for unusual but finite user settings.
            with np.errstate(over='ignore', invalid='ignore'):
                normalized = values[valid] / scale
            normalized = normalized * .25 + .5 if signed else normalized
            gray[valid] = np.rint(np.clip(normalized, 0, 1) * 255).astype(np.uint8)
            tile = cv.applyColorMap(gray, cv.COLORMAP_TURBO)[..., ::-1].copy()
            tile[~valid] = (255, 0, 255)
            if color_order == 'BGR':
                tile = tile[..., ::-1].copy()
            height = max(2, round(tile.shape[0] * width / tile.shape[1]))
            # Avoid very tall debug strips for portrait inputs.
            height = min(height, 480)
            canvas = np.zeros((height + 30, width, 3), np.uint8)
            canvas[30:] = cv.resize(tile, (width, height), interpolation=cv.INTER_NEAREST)
            cv.putText(canvas, title, (4, 20), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            tiles.append(canvas)
        strip = np.zeros((tiles[0].shape[0], base.shape[1], 3), np.uint8)
        for i, tile in enumerate(tiles):
            strip[:, i * width:(i + 1) * width] = tile
        return np.concatenate((base, strip), axis=0)


class RGBDBackgroundSegmenter:
    """Separate current from empty RGB-D at identical registered pixel positions.

    Color uses the boundary detector's encoded-Lab ellipse and optional declared
    background check/offset compensation. Depth is signed empty-current camera Z.
    Inputs are never modified. No registration, hole filling or smoothing occurs.
    """
    def __init__(self, *, background_chroma_tolerance=20., background_luma_tolerance=60.,
                 background_score_margin=.2, depth_threshold_m=.005, depth_margin_m=.001,
                 fusion_mode='union', missing_depth_policy='color', min_object_area=1,
                 background_compensate=False, background_check_p95=20., background_max_offset=40.):
        if fusion_mode not in ('union', 'depth_required'):
            raise ValueError('fusion_mode must be union or depth_required')
        if missing_depth_policy not in ('color', 'unknown'):
            raise ValueError('missing_depth_policy must be color or unknown')
        if type(min_object_area) is not int or min_object_area < 1:
            raise ValueError('min_object_area must be a positive integer')
        if type(background_compensate) is not bool:
            raise ValueError('background_compensate must be bool')
        numeric = dict(background_chroma_tolerance=background_chroma_tolerance,
                       background_luma_tolerance=background_luma_tolerance,
                       background_score_margin=background_score_margin,
                       depth_threshold_m=depth_threshold_m, depth_margin_m=depth_margin_m,
                       background_check_p95=background_check_p95, background_max_offset=background_max_offset)
        for key, value in numeric.items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f'{key} must be a finite number')
            try:
                numeric[key] = float(value)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f'{key} must be a finite number') from exc
            if not np.isfinite(numeric[key]) or numeric[key] < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        for key in ('background_chroma_tolerance', 'background_luma_tolerance',
                    'depth_threshold_m', 'background_check_p95', 'background_max_offset'):
            if numeric[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if numeric['background_score_margin'] > 1:
            raise ValueError('background_score_margin must be in [0,1]')
        if numeric['depth_margin_m'] >= numeric['depth_threshold_m']:
            raise ValueError('depth_margin_m must be less than depth_threshold_m')
        self.parameters = dict(numeric, fusion_mode=fusion_mode, missing_depth_policy=missing_depth_policy,
                               min_object_area=min_object_area, background_compensate=background_compensate)

    def detect(self, color, depth, *, background_color, background_depth,
               color_order, registered_to_depth=False, depth_scale_m,
               color_valid_mask=None, background_color_valid_mask=None,
               depth_valid_mask=None, background_depth_valid_mask=None,
               work_roi=None, background_check_mask=None, invalid_values=(), diagnostics=False):
        """Require both color uint8 HxWx3 and numeric optical-axis depth HxW pairs.

        All four images share one grid/view; both depths share depth_scale_m
        (metres/count, including floats). True validity masks declare real sensor
        support, not inpainted pixels. Nonpositive/nonfinite/sentinel depth is
        missing; no-depth is never zero difference. Check masks declare unchanged
        background in current coordinates. ROI is bool HxW, with no auto-alignment.
        """
        if registered_to_depth is not True:
            raise ValueError('registered_to_depth=True required for the aligned RGB-D pairs')
        if color_order not in ('RGB', 'BGR'):
            raise ValueError('explicit RGB or BGR required')
        if type(diagnostics) is not bool:
            raise ValueError('diagnostics must be bool')
        _validate_depth_scale(depth_scale_m)
        rgb = np.asarray(color)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError('color must be uint8 HxWx3')
        bg = np.asarray(background_color)
        if bg.dtype != np.uint8 or bg.shape != rgb.shape:
            raise ValueError('background_color must match current uint8 HxWx3')
        shape = rgb.shape[:2]
        def mask(value, name):
            return np.ones(shape, bool) if value is None else bool_mask(value, shape, name).copy()
        roi = mask(work_roi, 'work_roi')
        color_valid = mask(color_valid_mask, 'color_valid_mask') & mask(background_color_valid_mask, 'background_color_valid_mask')
        if background_check_mask is not None or self.parameters['background_compensate']:
            check = (np.zeros(shape, bool) if background_check_mask is None else
                     bool_mask(background_check_mask, shape, 'background_check_mask'))
            if (check & color_valid).sum() < 25:
                raise ValueError('background check needs at least 25 valid declared background pixels')
        current = _numeric_image(depth, 'depth')
        empty = _numeric_image(background_depth, 'background_depth')
        if current.shape != shape or empty.shape != shape:
            raise ValueError('both depth images must match the color grid')
        sentinels = _invalid_values(invalid_values)
        valid = mask(depth_valid_mask, 'depth_valid_mask') & mask(background_depth_valid_mask, 'background_depth_valid_mask')
        valid &= np.isfinite(current) & np.isfinite(empty) & (current > 0) & (empty > 0)
        valid &= ~np.isin(current, sentinels) & ~np.isin(empty, sentinels)
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            z = current * depth_scale_m
            z0 = empty * depth_scale_m
            delta = z0 - z
        valid &= np.isfinite(z) & np.isfinite(z0) & (z > 0) & (z0 > 0) & np.isfinite(delta)
        valid &= roi
        delta = np.where(valid, delta, np.nan)
        p = self.parameters
        cv = _opencv()
        rgb = rgb if color_order == 'RGB' else rgb[..., ::-1]
        lab = cv.cvtColor(rgb, cv.COLOR_RGB2LAB).astype(float)
        # Reuse exactly the boundary detector's background comparison. A rejected
        # reference disables color; it never silently falls back to border colors.
        cp = dict(p, background_difference_mode='separate_tolerances', background_shadow_mode='off',
                  color_distance=p['background_chroma_tolerance'], color_margin=0.)
        color_fg, color_uncertain, _, color_info, color_debug = _background_selection(
            rgb, lab, bg, color_order, color_valid, background_check_mask, (0, 0), cp)
        color_usable = color_valid & roi
        if color_info['status'] in ('rejected', 'insufficient'):
            color_usable[:] = False
            color_fg[:] = False
            color_uncertain[:] = True
            color_info.pop('fallback', None)
            color_info['selection_action'] = 'disable_color_evidence_keep_depth'
        color_info['difference_model']['applied_to_selection'] = bool(
            p['fusion_mode'] == 'union' and color_usable.any())
        color_fg &= color_usable
        t = p['depth_threshold_m']
        depth_fg = valid & (delta > t)
        farther = valid & (delta < -t)
        near_threshold = valid & (np.abs(np.abs(delta) - t) <= p['depth_margin_m'])
        conflict = color_fg & farther
        if p['fusion_mode'] == 'depth_required':
            obj = depth_fg.copy()
            observed = valid.copy()
        else:
            color_candidate = color_fg & ~farther
            if p['missing_depth_policy'] == 'unknown':
                color_candidate &= valid
                observed = valid.copy()
            else:
                observed = valid | color_usable
            obj = depth_fg | color_candidate
        # Uncertainty is orthogonal to selection. Color-only candidates, discarded
        # farther changes, absent modalities and threshold margins remain visible.
        uncertain = roi & (~valid | ~color_usable | color_uncertain | near_threshold |
                           farther | (color_fg & ~depth_fg))
        filtered = np.zeros(shape, bool)
        if p['min_object_area'] > 1:
            labels, parts = _components(obj)
            remove = np.zeros(len(parts) + 1, bool)
            for label, area in parts:
                if area < p['min_object_area']:
                    remove[label] = True
            filtered = remove[labels]
            obj &= ~filtered
            uncertain |= filtered
        status = ('empty_domain' if not roi.any() else 'insufficient' if not observed.any()
                  else 'candidate' if obj.any() else 'no_change_candidate')
        stages = dict(color_reference=color_info,
                      fusion=dict(status=status,
                                  rule='depth_foreground OR (color_foreground AND NOT farther)' if p['fusion_mode'] == 'union' else 'depth_foreground_only',
                                  warnings=['alignment_and_empty_reference_are_caller_assumptions',
                                            'color_only_change_can_be_shadow_or_thin_object',
                                            'no_change_does_not_prove_no_object',
                                            'depth_difference_is_camera_z_not_table_height',
                                            'real_image_accuracy_unvalidated']))
        metrics = dict(object_pixels=int(obj.sum()), uncertain_pixels=int(uncertain.sum()),
                       observed_pixels=int(observed.sum()), color_foreground_pixels=int(color_fg.sum()),
                       depth_foreground_pixels=int(depth_fg.sum()), conflict_pixels=int(conflict.sum()),
                       filtered_pixels=int(filtered.sum()))
        debug = {}
        if diagnostics:
            debug = dict(color_debug, depth_difference_m=delta, color_usable_mask=color_usable,
                         farther_mask=farther, depth_threshold_margin_mask=near_threshold,
                         color_only_mask=color_fg & ~depth_fg, filtered_mask=filtered, work_roi=roi)
        return RGBDBackgroundResult(obj, uncertain, observed, color_fg, depth_fg, valid, conflict,
                                    p.copy(), stages, metrics, debug)
