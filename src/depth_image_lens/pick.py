"""Supported overhead near-camera regions; no physical pinch-success inference."""
from dataclasses import dataclass, field
import numpy as np
from .debug import DebugResultMixin, _opencv
from .depth import _numeric_image, _validate_depth_scale, _invalid_values
from .geometry import bool_mask
from .outline import OuterContourResult


def _clearance(mask):
    """Euclidean distance to nearest excluded pixel center, including frame."""
    cv = _opencv()
    padded = np.pad(mask.astype(np.uint8), 1)
    return cv.distanceTransform(padded, cv.DIST_L2, cv.DIST_MASK_PRECISE)[1:-1, 1:-1]


@dataclass(frozen=True)
class PickCandidateResult(DebugResultMixin):
    state: str
    candidates: list
    point_xy: list | None
    contour_id: int | None
    reasons: list
    metrics: dict
    parameters: dict
    object_mask: np.ndarray = field(repr=False)
    candidate_mask: np.ndarray = field(repr=False)
    unknown_mask: np.ndarray = field(repr=False)
    excluded_mask: np.ndarray = field(repr=False)
    supported_mask: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)
    grasp_suitability: str = 'unverified'
    scope: str = 'supported_elevated_near_camera_regions_not_physical_pinch_success'

    def to_dict(self):
        return {k: v for k, v in vars(self).items()
                if not isinstance(v, np.ndarray) and k != 'diagnostics'}



class OverheadPickCandidateDetector:
    """Find geometric approach candidates inside one visible contour component.

    Assumes a fixed overhead camera and registered optical-axis current/empty Z.
    Background separation gates eligibility; absolute current Z ranks proximity.
    A broad elevated flat sheet may pass without being physically pinchable.
    """
    def __init__(self, *, min_elevation_m=.01, reference_error_m=.002,
                 boundary_guard_px=3., local_radius_px=2,
                 min_support_fraction=.8, local_depth_tolerance_m=.003,
                 min_region_area=25, min_region_radius_px=2., nearest_depth_band_m=.003):
        for name, value in dict(local_radius_px=local_radius_px, min_region_area=min_region_area).items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if local_radius_px > 10:
            raise ValueError('local_radius_px must be 1..10')
        numeric = dict(min_elevation_m=min_elevation_m, reference_error_m=reference_error_m,
                       boundary_guard_px=boundary_guard_px, min_support_fraction=min_support_fraction,
                       local_depth_tolerance_m=local_depth_tolerance_m,
                       min_region_radius_px=min_region_radius_px, nearest_depth_band_m=nearest_depth_band_m)
        for name, value in numeric.items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f'{name} must be a finite number')
            try:
                numeric[name] = float(value)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f'{name} must be a finite number') from exc
            if not np.isfinite(numeric[name]) or numeric[name] < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if min(numeric['min_elevation_m'], numeric['local_depth_tolerance_m']) <= 0:
            raise ValueError('positive elevation and local depth tolerance required')
        if not .5 < numeric['min_support_fraction'] <= 1:
            raise ValueError('min_support_fraction must be >.5 and <=1')
        self.parameters = dict(numeric, local_radius_px=local_radius_px, min_region_area=min_region_area)

    def detect(self, depth, outline, *, background_depth, contour_id=None,
               depth_scale_m, registered_to_mask=False, depth_valid_mask=None,
               background_valid_mask=None, segmentation_uncertain_mask=None,
               invalid_values=(), diagnostics=False):
        """Current and empty depth share the outline grid and metres/count scale.

        Outline must be freshly computed from the current object mask. Optional
        uncertainty/exclusion never changes contour identity. Missing/sentinel
        depth, reference errors and segmentation uncertainty are not filled.
        """
        if not isinstance(outline, OuterContourResult):
            raise ValueError('outline must be an OuterContourResult')
        if registered_to_mask is not True:
            raise ValueError('registered_to_mask=True required')
        if type(diagnostics) is not bool:
            raise ValueError('diagnostics must be bool')
        _validate_depth_scale(depth_scale_m)
        obj = outline.component_mask(contour_id)
        if contour_id is None:
            selected = outline.largest_contour()
            contour_id = None if selected is None else int(selected['id'])
        else:
            contour_id = int(contour_id)
        shape = obj.shape
        current = _numeric_image(depth, 'depth')
        empty = _numeric_image(background_depth, 'background_depth')
        if current.shape != shape or empty.shape != shape:
            raise ValueError('current/background depth must match the outline grid')
        def valid_mask(value, name):
            return np.ones(shape, bool) if value is None else bool_mask(value, shape, name).copy()
        valid = valid_mask(depth_valid_mask, 'depth_valid_mask') & valid_mask(background_valid_mask, 'background_valid_mask')
        sentinels = _invalid_values(invalid_values)
        valid &= obj & np.isfinite(current) & np.isfinite(empty) & (current > 0) & (empty > 0)
        valid &= ~np.isin(current, sentinels) & ~np.isin(empty, sentinels)
        uncertain = (np.zeros(shape, bool) if segmentation_uncertain_mask is None else
                     bool_mask(segmentation_uncertain_mask, shape, 'segmentation_uncertain_mask'))
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            z, z0 = current * depth_scale_m, empty * depth_scale_m
            height = z0 - z
        valid &= np.isfinite(z) & np.isfinite(z0) & (z > 0) & (z0 > 0) & np.isfinite(height)
        trusted = obj & ~uncertain
        valid &= trusted
        p = self.parameters
        boundary_clearance = _clearance(trusted)
        clearance = _clearance(valid)
        search = trusted & (boundary_clearance > p['boundary_guard_px'])
        interior = search & valid & (clearance > p['boundary_guard_px'])
        r = p['local_radius_px']; k = 2 * r + 1
        # Sliding views + bounded chunks avoid an H*W*k*k temporary image.
        zw = np.lib.stride_tricks.sliding_window_view(np.pad(np.where(valid, z, np.nan), r, constant_values=np.nan), (k, k))
        hw = np.lib.stride_tricks.sliding_window_view(np.pad(np.where(valid, height, np.nan), r, constant_values=np.nan), (k, k))
        local_z = np.full(shape, np.nan); local_h = np.full(shape, np.nan)
        support = np.zeros(shape, float); stable = np.zeros(shape, bool)
        chunk_rows = max(1, 1_000_000 // (shape[1] * k * k))
        for first in range(0, shape[0], chunk_rows):
            yy, xx = np.nonzero(interior[first:first + chunk_rows]); yy += first
            if not len(yy):
                continue
            a, b = zw[yy, xx], hw[yy, xx]
            mz, mh = np.nanmedian(a, axis=(1, 2)), np.nanmedian(b, axis=(1, 2))
            # A strict majority of the entire square must support BOTH current Z
            # and reference separation. Neighbors outside this component never count.
            with np.errstate(over='ignore', invalid='ignore'):
                good = ((np.abs(a - mz[:, None, None]) <= p['local_depth_tolerance_m']) &
                        (np.abs(b - mh[:, None, None]) <= p['local_depth_tolerance_m']))
                center_good = ((np.abs(z[yy, xx] - mz) <= p['local_depth_tolerance_m']) &
                               (np.abs(height[yy, xx] - mh) <= p['local_depth_tolerance_m']))
            fraction = good.mean(axis=(1, 2))
            local_z[yy, xx], local_h[yy, xx], support[yy, xx] = mz, mh, fraction
            stable[yy, xx] = center_good & (fraction >= p['min_support_fraction'])
        with np.errstate(over='ignore', invalid='ignore'):
            elevated = stable & (local_h - p['reference_error_m'] >= p['min_elevation_m'])
            elevated &= height - p['reference_error_m'] >= p['min_elevation_m']
        cv = _opencv()
        count, labels, stats, _ = cv.connectedComponentsWithStats(elevated.astype(np.uint8), connectivity=8)
        accepted = np.zeros(shape, bool); candidates = []; rejected = []
        for label in range(1, count):
            area = int(stats[label, cv.CC_STAT_AREA])
            if area < p['min_region_area']:
                rejected.append(dict(id=label, area_px=area, reasons=['region_too_small']))
                continue
            x0, y0, width, height_px = map(int, stats[label, :4])
            box = np.s_[y0:y0 + height_px, x0:x0 + width]
            region = labels[box] == label
            region_clearance = _clearance(region)
            points = region & (region_clearance > p['min_region_radius_px'])
            reasons = []
            if not points.any(): reasons.append('region_too_narrow')
            if reasons:
                rejected.append(dict(id=label, area_px=area, reasons=reasons))
                continue
            nearest = float(local_z[box][points].min())
            with np.errstate(over='ignore', invalid='ignore'):
                points &= local_z[box] - nearest <= p['nearest_depth_band_m']
            yy, xx = np.nonzero(points)
            yy_global, xx_global = yy + y0, xx + x0
            # Tie: greatest clearance, then topmost/leftmost pixel. No isolated
            # absolute minimum can win without support and candidate-region width.
            room = np.minimum(clearance[yy_global, xx_global], region_clearance[yy, xx])
            i = int(np.argmax(room)); y, x = int(yy_global[i]), int(xx_global[i])
            accepted[box] |= region
            candidates.append(dict(id=label, point_xy=[x, y], area_px=area,
                current_depth_m=float(z[y, x]), local_depth_m=float(local_z[y, x]),
                reference_separation_m=float(height[y, x]), local_separation_m=float(local_h[y, x]),
                local_support_fraction=float(support[y, x]), clearance_px=float(clearance[y, x]),
                region_clearance_px=float(region_clearance[yy[i], xx[i]]),
                nearest_supported_depth_m=nearest, grasp_suitability='unverified'))
        candidates.sort(key=lambda item: (item['local_depth_m'], -item['area_px'], item['point_xy'][1], item['point_xy'][0]))
        for rank, item in enumerate(candidates, 1): item['rank'] = rank
        unknown = (obj & uncertain) | (search & ~stable)
        reasons = []
        if candidates: state = 'candidate'
        elif not obj.any(): state = 'no_object'; reasons.append('no_visible_contour_component')
        elif not search.any(): state = 'unknown'; reasons.append('no_trusted_interior_with_clearance')
        elif unknown.any(): state = 'unknown'; reasons.append('missing_or_inconsistent_local_evidence')
        else: state = 'no_candidate'; reasons.append('no_region_meets_elevation_area_and_width')
        metrics = dict(object_pixels=int(obj.sum()), search_pixels=int(search.sum()),
                       pair_valid_pixels=int(valid.sum()), supported_pixels=int(stable.sum()),
                       candidate_pixels=int(accepted.sum()), unknown_pixels=int(unknown.sum()),
                       rejected_regions=rejected, reference='empty_minus_current_camera_z_m',
                       ranking='representative_local_current_z_ascending',
                       warnings=['overhead_registered_fixed_view_is_caller_assumption',
                                 'pixel_clearance_is_not_gripper_clearance',
                                 'elevated_flat_sheet_can_pass_without_pinch_access',
                                 'real_pick_success_and_real_image_accuracy_unvalidated'])
        # Point overlays are small and always present; dense diagnostics are opt-in.
        debug = dict(candidate_points_xy=np.array([v['point_xy'] for v in candidates], dtype=int).reshape(-1, 2))
        debug['selected_point_xy'] = np.array([candidates[0]['point_xy']] if candidates else [], dtype=int).reshape(-1, 2)
        if diagnostics:
            debug.update(depth_difference_m=np.where(valid, height, np.nan),
                         local_depth_m=local_z, local_separation_m=local_h,
                         support_fraction=support, clearance_px=clearance,
                         eligible_elevation_mask=elevated, pair_valid_mask=valid)
        return PickCandidateResult(state, candidates, candidates[0]['point_xy'].copy() if candidates else None,
                                   contour_id, reasons, metrics, p.copy(), obj, accepted, unknown,
                                   obj & ~search, stable, debug)
