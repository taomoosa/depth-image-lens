"""Fixed-view RGB-D empty-scene subtraction, with explicit evidence provenance."""
from dataclasses import dataclass, field
import numpy as np
from .boundary import _background_selection, _components
from .debug import DebugResultMixin, _opencv
from .depth import _numeric_image, _validate_depth_scale, _invalid_values
from .geometry import bool_mask
from ._rgbd_fusion import (evidence_fusion, color_seeded_fusion, DECISION_REASONS,
                           SEEDED_DECISION_REASONS, depth_anchored_fusion, ANCHORED_DECISION_REASONS)

from ._rgbd_color_region import color_region_fusion, REGION_DECISION_REASONS
from ._rgbd_color_supported import color_supported_fusion, SUPPORTED_DECISION_REASONS


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
    suppressed_mask: np.ndarray | None = field(default=None, repr=False)
    deferred_mask: np.ndarray | None = field(default=None, repr=False)
    confident_foreground_mask: np.ndarray | None = field(default=None, repr=False)
    confident_background_mask: np.ndarray | None = field(default=None, repr=False)
    unknown_mask: np.ndarray | None = field(default=None, repr=False)
    inferred_foreground_mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def outline_observed_mask(self):
        """Availability excluding unresolved decisions; does not remove all uncertainty."""
        observed = self.observed_mask.copy()
        if self.unknown_mask is not None:
            observed &= ~self.unknown_mask | self.object_mask
        for mask in (self.suppressed_mask, self.deferred_mask):
            if mask is not None:
                observed &= ~mask
        return observed

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
        output = np.concatenate((base, strip), axis=0)
        if 'fusion_decision' in self.diagnostics:
            # A dedicated view avoids losing decision evidence to the common
            # renderer's bounded number of mask thumbnails.
            palette = np.array([(35,35,35), (30,210,90), (50,160,245),
                                (50,220,220), (235,75,90), (180,100,220),
                                (240,150,40), (245,210,65)], dtype=np.uint8)
            tile = palette[self.diagnostics['fusion_decision']]
            if color_order == 'BGR':
                tile = tile[..., ::-1].copy()
            height = min(480, max(2, round(tile.shape[0] * width / tile.shape[1])))
            panel = np.zeros((max(height + 30, 212), base.shape[1], 3), np.uint8)
            panel[30:height + 30, :width] = cv.resize(tile, (width, height), interpolation=cv.INTER_NEAREST)
            cv.putText(panel, 'Fusion decision (not probability)', (4,20), cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            labels = ['0: no selected change / unavailable', '1: depth + reliable color',
                      '2: strong supported depth only', '3: chroma only (uncertain)',
                      '4: depth suppressed (unresolved)', '5: deferred (unresolved)',
                      '6: area filtered']
            if self.parameters['fusion_mode'] == 'color_seeded':
                labels = ['0: no selected change / unavailable', '1: strict seed + depth',
                          '2: unused in this mode', '3: strict seed (color only)',
                          '4: depth suppressed (unresolved)', '5: deferred (unresolved)',
                          '6: area filtered', '7: bounded growth (uncertain)']
            elif self.parameters['fusion_mode'] == 'color_supported':
                labels = ['0: no change / unavailable', '1: color + supported depth seed',
                          '2: strong depth seed', '3: unsupported color (unresolved)',
                          '4: unsupported depth', '5: unresolved local evidence',
                          '6: area filtered', '7: seed-color growth (uncertain)']
            elif self.parameters['fusion_mode'] == 'color_region':
                labels = ['0: no color candidate', '1: region accepted + raw depth',
                          '2: unused', '3: region depth unsupported',
                          '4: raw depth unselected', '5: region insufficient evidence',
                          '6: area filtered', '7: whole-region inference (uncertain)']
            elif self.parameters['fusion_mode'] == 'depth_anchored':
                labels = ['0: no change / unavailable', '1: supported depth anchor',
                          '2: thin depth anchor', '3: background anchor',
                          '4: raw depth unresolved', '5: boundary/color unresolved',
                          '6: area filtered', '7: RGB refinement (uncertain)']
            for i, label in enumerate(labels):
                ink = palette[i] if color_order == 'RGB' else palette[i, ::-1]
                if i == 0:
                    ink = (210, 210, 210)
                cv.putText(panel, label, (width + 8, 24 + 22*i), cv.FONT_HERSHEY_SIMPLEX,.4,tuple(map(int,ink)),1)
            output = np.concatenate((output, panel), axis=0)
        return output


class RGBDBackgroundSegmenter:
    """Separate current from empty RGB-D at identical registered pixel positions.

    Color uses the boundary detector's encoded-Lab ellipse and optional declared
    background check/offset compensation. Depth is signed empty-current camera Z.
    Inputs are never modified. No registration or depth filling occurs.
    Optional color_seeded mode adds bounded, evidence-gated mask growth.
    depth_anchored mode refines depth-supported geometry with current RGB edges;
    missing depth remains unknown regardless of missing_depth_policy.
    color_supported validates every seed with local depth and grows source colors
    only within a bounded neighborhood; it also never fills missing depth.
    """
    def __init__(self, *, background_chroma_tolerance=20., background_luma_tolerance=60.,
                 background_score_margin=.2, depth_threshold_m=.005, depth_margin_m=.001,
                 fusion_mode='union', missing_depth_policy='color', min_object_area=1,
                 background_compensate=False, background_check_p95=20., background_max_offset=40.,
                 strong_depth_ratio=3., depth_support_radius_px=1,
                 depth_support_fraction=.25, depth_support_tolerance_m=.003,
                 color_relaxation_ratio=.5, color_growth_max_steps=2,
                 anchored_color_sigma=12., anchored_pairwise_weight=2.,
                 anchored_refine_radius_px=4, anchored_iterations=24, anchored_anchor_ratio=2.,
                 supported_color_tolerance=18., supported_growth_radius_px=4,
                 supported_min_seed_area=3, supported_min_depth_fraction=.1,
                 supported_anchor_ratio=2., supported_relaxation_ratio=.2,
                 region_color_tolerance=18., region_color_relaxation_ratio=.2,
                 region_min_area=32, region_max_color_groups=64,
                 region_min_valid_fraction=.5, region_mean_depth_ratio=1.,
                 region_min_depth_fraction=.25, region_depth_clip_ratio=3.):
        if fusion_mode not in ('union', 'depth_required', 'evidence', 'color_seeded', 'depth_anchored', 'color_supported', 'color_region'):
            raise ValueError('fusion_mode must be union, depth_required, evidence, color_seeded, depth_anchored, color_supported or color_region')
        for key, value, lower, upper in (('anchored_refine_radius_px', anchored_refine_radius_px, 0, 20),
                                          ('anchored_iterations', anchored_iterations, 1, 100),
                                          ('supported_growth_radius_px', supported_growth_radius_px, 0, 20)):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{key} must be an integer in {lower}..{upper}')
        for key, value in (('region_min_area', region_min_area), ('region_max_color_groups', region_max_color_groups)):
            if type(value) is not int or value < 1 or (key == 'region_max_color_groups' and value > 256):
                raise ValueError(f'{key} must be a positive integer (color group cap <=256)')
        if type(supported_min_seed_area) is not int or supported_min_seed_area < 1:
            raise ValueError('supported_min_seed_area must be a positive integer')
        if type(color_growth_max_steps) is not int or not 0 <= color_growth_max_steps <= 20:
            raise ValueError('color_growth_max_steps must be an integer in 0..20')
        if type(depth_support_radius_px) is not int or not 1 <= depth_support_radius_px <= 5:
            raise ValueError('depth_support_radius_px must be an integer in 1..5')
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
                       background_check_p95=background_check_p95, background_max_offset=background_max_offset,
                       strong_depth_ratio=strong_depth_ratio, depth_support_fraction=depth_support_fraction,
                       depth_support_tolerance_m=depth_support_tolerance_m,
                       color_relaxation_ratio=color_relaxation_ratio,
                       anchored_color_sigma=anchored_color_sigma,
                       anchored_pairwise_weight=anchored_pairwise_weight,
                       anchored_anchor_ratio=anchored_anchor_ratio,
                       supported_color_tolerance=supported_color_tolerance,
                       supported_min_depth_fraction=supported_min_depth_fraction,
                       supported_anchor_ratio=supported_anchor_ratio,
                       supported_relaxation_ratio=supported_relaxation_ratio,
                       region_color_tolerance=region_color_tolerance,
                       region_color_relaxation_ratio=region_color_relaxation_ratio,
                       region_min_valid_fraction=region_min_valid_fraction,
                       region_mean_depth_ratio=region_mean_depth_ratio,
                       region_min_depth_fraction=region_min_depth_fraction,
                       region_depth_clip_ratio=region_depth_clip_ratio)
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
        for key in ('region_min_valid_fraction', 'region_min_depth_fraction', 'region_color_relaxation_ratio'):
            if not 0 < numeric[key] <= 1:
                raise ValueError(f'{key} must be in (0,1]')
        if not 0 < numeric['region_color_tolerance'] <= 441:
            raise ValueError('region_color_tolerance must be in (0,441]')
        if not 0 < numeric['region_mean_depth_ratio'] < numeric['region_depth_clip_ratio'] <= 100:
            raise ValueError('require 0 < region_mean_depth_ratio < region_depth_clip_ratio <= 100')
        if not 0 < numeric['supported_color_tolerance'] <= 441:
            raise ValueError('supported_color_tolerance must be in (0,441]')
        for key in ('supported_min_depth_fraction', 'supported_relaxation_ratio'):
            if not 0 < numeric[key] <= 1:
                raise ValueError(f'{key} must be in (0,1]')
        if not 1 <= numeric['supported_anchor_ratio'] <= 4:
            raise ValueError('supported_anchor_ratio must be in [1,4]')
        if not 0 < numeric['anchored_color_sigma'] <= 441:
            raise ValueError('anchored_color_sigma must be in (0,441]')
        if numeric['anchored_pairwise_weight'] > 100:
            raise ValueError('anchored_pairwise_weight must be in [0,100]')
        if not 1 <= numeric['anchored_anchor_ratio'] <= 4:
            raise ValueError('anchored_anchor_ratio must be in [1,4]')
        if numeric['background_score_margin'] > 1:
            raise ValueError('background_score_margin must be in [0,1]')
        if numeric['depth_margin_m'] >= numeric['depth_threshold_m']:
            raise ValueError('depth_margin_m must be less than depth_threshold_m')
        if numeric['strong_depth_ratio'] < 1:
            raise ValueError('strong_depth_ratio must be >= 1')
        if not 0 < numeric['depth_support_fraction'] <= 1:
            raise ValueError('depth_support_fraction must be in (0,1]')
        if not 0 < numeric['color_relaxation_ratio'] <= 1:
            raise ValueError('color_relaxation_ratio must be in (0,1]')
        self.parameters = dict(numeric, fusion_mode=fusion_mode, missing_depth_policy=missing_depth_policy,
                               min_object_area=min_object_area, background_compensate=background_compensate,
                               depth_support_radius_px=depth_support_radius_px,
                               color_growth_max_steps=color_growth_max_steps,
                               anchored_refine_radius_px=anchored_refine_radius_px,
                               anchored_iterations=anchored_iterations,
                               supported_growth_radius_px=supported_growth_radius_px,
                               supported_min_seed_area=supported_min_seed_area,
                               region_min_area=region_min_area, region_max_color_groups=region_max_color_groups)

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
        current_color_valid = mask(color_valid_mask, 'color_valid_mask')
        color_valid = current_color_valid & mask(background_color_valid_mask, 'background_color_valid_mask')
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
            color_info['selection_action'] = ('disable_color_require_strong_supported_depth'
                if p['fusion_mode'] == 'evidence' else 'disable_color_no_seeds'
                if p['fusion_mode'] == 'color_seeded' else 'disable_reference_difference_keep_current_rgb_edges'
                if p['fusion_mode'] == 'depth_anchored' else 'disable_color_evidence_keep_depth')
        color_info['difference_model']['applied_to_selection'] = bool(
            p['fusion_mode'] != 'depth_required' and color_usable.any())
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
        suppressed = np.zeros(shape, bool)
        deferred = np.zeros(shape, bool)
        fusion_debug = {}
        if p['fusion_mode'] == 'evidence':
            obj, suppressed, deferred, fusion_debug = evidence_fusion(
                delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
                color_debug['background_chroma_distance'], valid, p)
            uncertain |= suppressed | deferred | fusion_debug['accepted_depth_only_mask']
        elif p['fusion_mode'] == 'color_seeded':
            obj, suppressed, deferred, fusion_debug = color_seeded_fusion(
                delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
                color_debug, valid, p)
            uncertain |= suppressed | deferred | fusion_debug['accepted_growth_mask']
        elif p['fusion_mode'] == 'depth_anchored':
            obj, suppressed, deferred, fusion_debug = depth_anchored_fusion(
                delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
                lab, current_color_valid & roi, color_debug, valid, roi, p)
            observed = valid.copy()
            uncertain = fusion_debug['unknown_mask'].copy()
        elif p['fusion_mode'] == 'color_supported':
            obj, suppressed, deferred, fusion_debug = color_supported_fusion(
                delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
                lab, current_color_valid & roi, color_debug, valid, roi, p)
            observed = valid.copy()
            uncertain |= fusion_debug['unknown_mask']
            fusion_debug['unknown_mask'] = uncertain.copy()
        elif p['fusion_mode'] == 'color_region':
            obj, suppressed, deferred, fusion_debug = color_region_fusion(
                delta, depth_fg, farther, color_fg, color_usable, color_uncertain,
                lab, current_color_valid & roi, color_debug, valid, roi, p)
            observed = valid.copy()
            uncertain |= fusion_debug['unknown_mask']
            fusion_debug['unknown_mask'] = uncertain.copy()
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
            if fusion_debug:
                fusion_debug['fusion_decision'][filtered] = 6
                for key in ('accepted_both_mask', 'accepted_depth_only_mask', 'accepted_color_only_mask',
                            'accepted_growth_mask'):
                    if key in fusion_debug:
                        fusion_debug[key] &= ~filtered
        if p['fusion_mode'] == 'color_supported':
            for key in ('supported_seed_mask', 'color_seed_mask', 'depth_seed_mask'):
                fusion_debug[key] &= ~filtered
        if p['fusion_mode'] in ('depth_anchored', 'color_supported', 'color_region'):
            fusion_debug['inferred_foreground_mask'] &= ~filtered
            fusion_debug['confident_foreground_mask'] &= ~filtered
            fusion_debug['unknown_mask'] |= filtered
        status = ('empty_domain' if not roi.any() else 'insufficient' if not observed.any()
                  else 'candidate' if obj.any() else 'unresolved' if (suppressed | deferred).any()
                  else 'no_change_candidate')
        stages = dict(color_reference=color_info,
                      fusion=dict(status=status,
                                  rule={'union': 'depth_foreground OR (color_foreground AND NOT farther)',
                                        'depth_required': 'depth_foreground_only',
                                        'evidence': 'color_confirmation_or_strong_supported_depth_or_chroma_candidate',
                                        'color_seeded': 'strict_chroma_seeds_plus_bounded_relaxed_color_supported_depth_growth',
                                        'depth_anchored': 'depth_anchors_and_bounded_full_lab_diffusion_refinement',
                                        'color_supported': 'depth_validated_seeds_and_bounded_local_seed_color_growth',
                                        'color_region': 'connected_full_lab_regions_with_clipped_mean_and_depth_density'}[p['fusion_mode']],
                                  warnings=['alignment_and_empty_reference_are_caller_assumptions',
                                            'color_only_change_can_be_shadow_or_thin_object',
                                            'no_change_does_not_prove_no_object',
                                            'depth_difference_is_camera_z_not_table_height',
                                            'real_image_accuracy_unvalidated']))
        metrics = dict(object_pixels=int(obj.sum()), uncertain_pixels=int(uncertain.sum()),
                       observed_pixels=int(observed.sum()), color_foreground_pixels=int(color_fg.sum()),
                       depth_foreground_pixels=int(depth_fg.sum()), conflict_pixels=int(conflict.sum()),
                       filtered_pixels=int(filtered.sum()))
        if p['fusion_mode'] in ('evidence', 'color_seeded', 'depth_anchored', 'color_supported', 'color_region'):
            reasons = (REGION_DECISION_REASONS if p['fusion_mode'] == 'color_region' else
                       SUPPORTED_DECISION_REASONS if p['fusion_mode'] == 'color_supported' else
                       ANCHORED_DECISION_REASONS if p['fusion_mode'] == 'depth_anchored' else
                       SEEDED_DECISION_REASONS if p['fusion_mode'] == 'color_seeded' else DECISION_REASONS)
            stages['fusion']['decision_reasons'] = {str(code): reason for code, reason in reasons.items()}
            stages['fusion']['warnings'] += ['spatial_support_is_not_temporal_stability_or_probability',
                                             'suppressed_and_deferred_pixels_are_not_confirmed_background']
            metrics.update(suppressed_pixels=int(suppressed.sum()), deferred_pixels=int(deferred.sum()),
                           decision_counts={reason: int((fusion_debug['fusion_decision'] == code).sum())
                                            for code, reason in reasons.items()})
        if p['fusion_mode'] == 'color_seeded':
            stages['fusion']['warnings'] += [
                'chroma_seeds_required_same_color_and_luma_only_objects_can_be_lost',
                'growth_is_uncertain_inference_not_observed_material',
                'adjacent_shadow_with_depth_error_can_grow_within_step_limit',
                'color_misalignment_and_colored_shadows_can_seed_false_objects']
            metrics.update(seed_pixels=int(fusion_debug['color_seed_mask'].sum()),
                           grown_pixels=int(fusion_debug['accepted_growth_mask'].sum()))
        if p['fusion_mode'] == 'depth_anchored':
            stages['fusion']['warnings'] += [
                'confidence_is_heuristic_evidence_not_calibrated_probability_or_material_truth',
                'coherent_depth_bias_and_corrupted_holes_can_pass',
                'refinement_is_local_not_global_optimum_or_registration_correction',
                'missing_depth_is_always_unknown_in_this_mode',
                'inferred_geometry_is_not_measured_depth_or_safe_grasp_support']
            metrics.update({key + '_pixels': int(fusion_debug[key + '_mask'].sum())
                            for key in ('confident_foreground', 'confident_background', 'unknown',
                                        'inferred_foreground', 'foreground_anchor', 'thin_depth_anchor')})
            for key in ('refinement_iterations', 'refinement_converged', 'refinement_max_update'):
                metrics[key] = fusion_debug[key]
            if not fusion_debug['refinement_converged']:
                stages['fusion']['warnings'].append('refinement_iteration_budget_reached')
        if p['fusion_mode'] == 'color_supported':
            stages['fusion']['warnings'] += [
                'color_only_candidates_require_local_depth_supported_seeds',
                'growth_is_bounded_local_full_lab_inference_not_depth_measurement',
                'missing_depth_is_always_unknown_in_this_mode',
                'coherent_depth_bias_and_similar_colored_touching_shadows_can_pass',
                'unseeded_thin_or_shallow_objects_can_be_lost',
                'confidence_is_heuristic_not_physical_truth_or_grasp_safety']
            metrics.update({key + '_pixels': int(fusion_debug[key + '_mask'].sum())
                            for key in ('supported_seed', 'color_seed', 'depth_seed',
                                        'inferred_foreground', 'unsupported_color', 'support_rejected')})
        if p['fusion_mode'] == 'color_region':
            stages['fusion']['regions'] = fusion_debug['region_statistics']
            stages['fusion']['warnings'] += [
                'whole_region_support_can_accept_same_color_touching_shadows_and_holes',
                'inferred_pixels_are_not_measured_depth_or_safe_grasp_support',
                'missing_depth_is_never_filled',
                'coherent_depth_bias_can_pass_region_mean_and_density',
                'fixed_color_prototypes_can_split_gradients_and_thin_regions']
            metrics['color_region_count'] = len(fusion_debug['region_statistics'])
            metrics['color_region_budget_exceeded'] = fusion_debug['region_color_budget_exceeded']
        debug = {}
        if diagnostics:
            debug = dict(color_debug, **fusion_debug, depth_difference_m=delta, color_usable_mask=color_usable,
                         farther_mask=farther, depth_threshold_margin_mask=near_threshold,
                         color_only_mask=color_fg & ~depth_fg, filtered_mask=filtered, work_roi=roi)
        return RGBDBackgroundResult(obj, uncertain, observed, color_fg, depth_fg, valid, conflict,
                                    p.copy(), stages, metrics, debug,
                                    suppressed_mask=suppressed, deferred_mask=deferred,
                                    **{key: fusion_debug.get(key) for key in (
                                        'confident_foreground_mask', 'confident_background_mask',
                                        'unknown_mask', 'inferred_foreground_mask')})
