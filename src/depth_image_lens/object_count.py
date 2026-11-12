"""Visible color-identity hypotheses from overhead RGB-D or external masks.

This is not semantic instance segmentation. Color distinctness, registration,
visibility, and an optional upper camera-Z band are caller assumptions.
"""
from dataclasses import dataclass, field
import numpy as np
from .background import RGBDBackgroundSegmenter
from .debug import DebugResultMixin, _opencv
from .geometry import bool_mask
from .depth import _numeric_image, _validate_depth_scale, _invalid_values


@dataclass(frozen=True)
class ObjectCountResult(DebugResultMixin):
    count: int | None
    candidate_count: int
    state: str
    groups: list
    reasons: list
    warnings: list
    metrics: dict
    parameters: dict
    label_map: np.ndarray = field(repr=False)
    object_mask: np.ndarray = field(repr=False)
    candidate_mask: np.ndarray = field(repr=False)
    uncertain_mask: np.ndarray = field(repr=False)
    excluded_mask: np.ndarray = field(repr=False)
    rejected_mask: np.ndarray = field(repr=False)
    observed_mask: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)
    scope: str = 'visible_distinguishable_color_groups_not_verified_physical_objects'
    validation_status: str = 'synthetic_tests_only_real_image_accuracy_unvalidated'

    def to_dict(self):
        """JSON-safe summary; full-resolution masks/labels stay in the result."""
        return {key: value for key, value in vars(self).items()
                if not isinstance(value, np.ndarray) and key != 'diagnostics'}

    def group_mask(self, group_id):
        """Return a new bool mask for a current-frame group; IDs are not tracked."""
        if type(group_id) is not int or not 1 <= group_id <= self.candidate_count:
            raise ValueError('group_id must identify a current result group')
        return self.label_map == group_id

    def render_debug(self, color=None, *, color_order='RGB', depth=None,
                     depth_scale_m=.001, mask_alpha=.35, panel=True, max_lines=32,
                     group_labels=True):
        """Show color-coded group IDs above the independent evidence-mask tiles.

        Display colors are deterministic by result-local ID, not measured RGB or
        tracked object identities. The complete legend survives panel=False and
        max_lines; group_labels=False hides only the spatial ID badges. Candidate
        groups are still shown when count is None. No detection is rerun.
        """
        from .object_count_debug import render_object_count_debug
        return render_object_count_debug(
            self, color, color_order=color_order, depth=depth,
            depth_scale_m=depth_scale_m, mask_alpha=mask_alpha, panel=panel,
            max_lines=max_lines, group_labels=group_labels)


@dataclass
class _CountEvidence:
    """Internal common evidence, never a fabricated segmentation result."""
    object_mask: np.ndarray
    color_usable_mask: np.ndarray
    depth_valid_mask: np.ndarray
    depth_foreground_mask: np.ndarray
    depth_threshold_margin_mask: np.ndarray
    uncertain_mask: np.ndarray
    conflict_mask: np.ndarray
    unresolved_mask: np.ndarray
    filtered_mask: np.ndarray
    depth_difference_m: np.ndarray
    color_reference_status: str
    parameters: dict
    summary: dict | None = None
    color_foreground_mask: np.ndarray | None = None
    input_uncertain_mask: np.ndarray | None = None
    cleanup_changed_mask: np.ndarray | None = None


def _chromaticity(rgb):
    values = rgb.astype(np.float64)
    return 255. * values / np.maximum(values.sum(axis=-1, keepdims=True), 1.)


def _dilate(mask, radius):
    if not radius:
        return mask.copy()
    cv = _opencv()
    return cv.dilate(mask.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                     borderType=cv.BORDER_CONSTANT, borderValue=0).astype(bool)


def _palette(features, tolerance, max_groups):
    """Bounded deterministic histogram modes, without single-linkage chains.

    Bins are narrower than the matching radius; sub-unit tolerances use exact
    feature bins, including extremely tiny finite tolerances. Full-precision
    features are used for distances and assignment. No random pixel sampling.
    """
    width = min(4., tolerance / 4.) if tolerance >= 1. else 0.
    quantized = np.rint(features / width).astype(np.int16) if width else features
    unique, inverse, weights = np.unique(quantized, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)  # Stable across NumPy unique(axis=...) versions.
    means = (np.column_stack([np.bincount(inverse, weights=features[:, i]) / weights
                              for i in range(3)]) if width else unique)
    remaining = np.ones(len(weights), bool)
    centers = []
    for _ in range(max_groups):
        if not remaining.any():
            break
        seed = int(np.argmax(np.where(remaining, weights, 0)))
        near = remaining & (np.linalg.norm(means - means[seed], axis=1) <= tolerance)
        center = means[seed] + np.average(means[near] - means[seed], weights=weights[near], axis=0)
        # Membership is always bounded around one center, never transitively
        # expanded through intermediate colors.
        near = remaining & (np.linalg.norm(means - center, axis=1) <= tolerance)
        centers.append(center)
        remaining[near] = False
    return np.asarray(centers).reshape(-1, 3), bool(remaining.any())


def _assign(features, centers, tolerance, margin):
    labels = np.zeros(len(features), np.int32)
    ambiguous = np.zeros(len(features), bool)
    if not len(centers):
        return labels, ambiguous
    # Bound the N x K distance temporary to about one million doubles.
    chunk = max(1, 1_000_000 // len(centers))
    for start in range(0, len(features), chunk):
        values = features[start:start+chunk]
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        first = distances[np.arange(len(values)), nearest]
        second = (np.partition(distances, 1, axis=1)[:, 1]
                  if len(centers) > 1 else np.full(len(values), np.inf))
        good = first <= tolerance
        ambiguous[start:start+chunk] = good & (second - first < margin)
        labels[start:start+chunk] = np.where(good & (second-first >= margin), nearest+1, 0)
    return labels, ambiguous


class OverheadObjectCounter:
    """Count supported, near-solid visible color groups across disconnected parts.

    Proportional scaling of encoded RGB leaves the color feature unchanged; real
    shadows/white-balance changes need not obey that model. Brightness-only color
    differences can merge physical objects into one accepted color-group estimate.
    No hidden-object count is inferred.
    Requires the optional OpenCV extra for either detection entrypoint, not on import.
    """
    def __init__(self, *, background_segmenter=None, color_tolerance=18.,
                 color_margin=4., min_object_area=64, min_fragment_area=8,
                 min_depth_support_fraction=.1, exclusion_guard_px=2,
                 max_reference_separation_m=None, max_color_groups=32,
                 min_brightness=24., min_saturation=.12, stability_fraction=.2):
        if background_segmenter is not None and not isinstance(background_segmenter, RGBDBackgroundSegmenter):
            raise ValueError('background_segmenter must be RGBDBackgroundSegmenter or None')
        for name, value in dict(min_object_area=min_object_area, min_fragment_area=min_fragment_area,
                                max_color_groups=max_color_groups, exclusion_guard_px=exclusion_guard_px).items():
            if type(value) is not int or value < (0 if name == 'exclusion_guard_px' else 1):
                raise ValueError(f'{name} must be an integer in its valid range')
        if max_color_groups > 128 or exclusion_guard_px > 64:
            raise ValueError('max_color_groups must be <=128 and exclusion_guard_px <=64')
        if min_fragment_area > min_object_area:
            raise ValueError('min_fragment_area must be <= min_object_area')
        numeric = dict(color_tolerance=color_tolerance, color_margin=color_margin,
                       min_depth_support_fraction=min_depth_support_fraction,
                       min_brightness=min_brightness, min_saturation=min_saturation,
                       stability_fraction=stability_fraction)
        if max_reference_separation_m is not None:
            numeric['max_reference_separation_m'] = max_reference_separation_m
        for name, value in numeric.items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f'{name} must be a finite number')
            try:
                numeric[name] = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{name} must be a finite number') from exc
            if not np.isfinite(numeric[name]) or numeric[name] < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < numeric['color_tolerance'] <= 128 or numeric['color_margin'] >= numeric['color_tolerance']:
            raise ValueError('color_tolerance must be in (0,128] and color_margin less than it')
        if not 0 < numeric['min_depth_support_fraction'] <= 1:
            raise ValueError('min_depth_support_fraction must be in (0,1]')
        if not 0 <= numeric['min_saturation'] <= 1 or not 0 < numeric['min_brightness'] <= 255:
            raise ValueError('min_saturation must be in [0,1], min_brightness in (0,255]')
        if not 0 < numeric['stability_fraction'] <= .5:
            raise ValueError('stability_fraction must be in (0,.5]')
        if max_reference_separation_m is not None and numeric['max_reference_separation_m'] <= 0:
            raise ValueError('max_reference_separation_m must be positive or None')
        numeric.setdefault('max_reference_separation_m', None)
        self.parameters = dict(numeric, min_object_area=min_object_area, min_fragment_area=min_fragment_area,
                               exclusion_guard_px=exclusion_guard_px, max_color_groups=max_color_groups)
        self.background_segmenter = background_segmenter or RGBDBackgroundSegmenter()

    def detect(self, color, depth, *, background_color, background_depth, color_order,
               depth_scale_m, registered_to_depth=False, arm_mask=None, exclusion_mask=None,
               work_roi=None, color_valid_mask=None, background_color_valid_mask=None,
               depth_valid_mask=None, background_depth_valid_mask=None,
               background_check_mask=None, invalid_values=(), diagnostics=False):
        """All four images must share a fixed, registered grid (including a crop).

        Depth is optical-axis Z; depth_scale_m is mandatory for every dtype.
        True arm/exclusion pixels and a square guard are omitted, not called empty.
        An optional upper separation band is empty-current Z, NOT table height.
        Missing depth never establishes that a pixel is below this band.
        """
        if type(diagnostics) is not bool:
            raise ValueError('diagnostics must be bool')
        segmentation = self.background_segmenter.detect(
            color, depth, background_color=background_color, background_depth=background_depth,
            color_order=color_order, registered_to_depth=registered_to_depth, depth_scale_m=depth_scale_m,
            work_roi=work_roi, color_valid_mask=color_valid_mask,
            background_color_valid_mask=background_color_valid_mask, depth_valid_mask=depth_valid_mask,
            background_depth_valid_mask=background_depth_valid_mask, background_check_mask=background_check_mask,
            invalid_values=invalid_values, diagnostics=True)
        evidence = _CountEvidence(
            segmentation.object_mask, segmentation.diagnostics['color_usable_mask'],
            segmentation.depth_valid_mask, segmentation.depth_foreground_mask,
            segmentation.diagnostics['depth_threshold_margin_mask'], segmentation.uncertain_mask,
            segmentation.conflict_mask, segmentation.suppressed_mask | segmentation.deferred_mask,
            segmentation.diagnostics['filtered_mask'], segmentation.diagnostics['depth_difference_m'],
            segmentation.stages['color_reference']['status'],
            dict(input_mode='rgbd_background', background_segmenter=segmentation.parameters.copy()),
            segmentation.to_dict(), segmentation.color_foreground_mask)
        return self._count(color, evidence, color_order=color_order, depth_scale_m=depth_scale_m,
                           arm_mask=arm_mask, exclusion_mask=exclusion_mask, work_roi=work_roi,
                           diagnostics=diagnostics)

    def detect_mask(self, color, object_mask, *, color_order, depth=None, background_depth=None,
                    depth_scale_m=None, registered_to_depth=False, arm_mask=None, exclusion_mask=None,
                    work_roi=None, color_valid_mask=None, depth_valid_mask=None,
                    background_depth_valid_mask=None, observed_mask=None, uncertain_mask=None,
                    cleanup_changed_mask=None, invalid_values=(), diagnostics=False):
        """Group an existing bool mask without running background segmentation.

        Color is uint8 HxWx3; every mask/depth shares that grid. Optional current
        and empty depth must be supplied together, with explicit scale and
        registered_to_depth=True. Their signed camera-Z support uses the configured
        background segmenter's depth threshold/margin ONLY, never its segmentation.
        Without depth, color hypotheses are returned but count is always None.

        observed_mask=False means unknown coverage, not known background. Carry
        upstream uncertainty in uncertain_mask and ALL cleanup changes (including
        removals) in cleanup_changed_mask. These masks block an accepted count in
        the searched domain and never create depth evidence. Mask selection and
        completeness remain caller assumptions; no RGB background is required.
        """
        if type(diagnostics) is not bool or type(registered_to_depth) is not bool:
            raise ValueError('diagnostics and registered_to_depth must be bool')
        if color_order not in ('RGB', 'BGR'):
            raise ValueError('explicit RGB or BGR required')
        rgb = np.asarray(color)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
            raise ValueError('color must be uint8 HxWx3')
        shape = rgb.shape[:2]
        def mask(value, name, default=False):
            return np.full(shape, default, bool) if value is None else bool_mask(value, shape, name).copy()
        candidate = bool_mask(object_mask, shape, 'object_mask').copy()
        roi = mask(work_roi, 'work_roi', True)
        available = mask(observed_mask, 'observed_mask', True) & roi
        usable_color = mask(color_valid_mask, 'color_valid_mask', True) & available
        input_uncertain = mask(uncertain_mask, 'uncertain_mask')
        changed = mask(cleanup_changed_mask, 'cleanup_changed_mask')
        sentinels = _invalid_values(invalid_values)
        has_depth = depth is not None or background_depth is not None
        if has_depth and (depth is None or background_depth is None):
            raise ValueError('depth and background_depth must be supplied together')
        bp = self.background_segmenter.parameters
        threshold, margin = bp['depth_threshold_m'], bp['depth_margin_m']
        if has_depth:
            if registered_to_depth is not True:
                raise ValueError('registered_to_depth=True required for the aligned RGB-D inputs')
            _validate_depth_scale(depth_scale_m)
            current = _numeric_image(depth, 'depth')
            empty = _numeric_image(background_depth, 'background_depth')
            if current.shape != shape or empty.shape != shape:
                raise ValueError('both depth images must match the color grid')
            valid = mask(depth_valid_mask, 'depth_valid_mask', True)
            valid &= mask(background_depth_valid_mask, 'background_depth_valid_mask', True)
            valid &= available & np.isfinite(current) & np.isfinite(empty) & (current > 0) & (empty > 0)
            valid &= ~np.isin(current, sentinels) & ~np.isin(empty, sentinels)
            with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                z, z0 = current * depth_scale_m, empty * depth_scale_m
                delta = z0 - z
            valid &= np.isfinite(z) & np.isfinite(z0) & (z > 0) & (z0 > 0) & np.isfinite(delta)
            delta = np.where(valid, delta, np.nan)
        else:
            if (depth_scale_m is not None or depth_valid_mask is not None
                    or background_depth_valid_mask is not None or len(sentinels)):
                raise ValueError('depth scale, validity and sentinels require a current/background depth pair')
            valid = np.zeros(shape, bool)
            delta = np.full(shape, np.nan)
        near_threshold = valid & (np.abs(np.abs(delta) - threshold) <= margin)
        depth_foreground = valid & (delta > threshold) & ~input_uncertain & ~changed
        conflict = candidate & valid & (delta < -threshold)
        uncertain = input_uncertain | changed | ~available | ~usable_color | ~valid | near_threshold | conflict
        evidence = _CountEvidence(
            candidate, usable_color, valid, depth_foreground, near_threshold,
            uncertain, conflict, np.zeros(shape, bool), np.zeros(shape, bool), delta,
            'not_used_external_mask',
            dict(input_mode='external_mask', background_segmenter=None,
                 depth_evidence='measured_current_empty_pair' if has_depth else 'unavailable',
                 depth_support_parameters=dict(depth_threshold_m=threshold, depth_margin_m=margin)),
            input_uncertain_mask=input_uncertain, cleanup_changed_mask=changed)
        return self._count(color, evidence, color_order=color_order, depth_scale_m=depth_scale_m,
                           arm_mask=arm_mask, exclusion_mask=exclusion_mask, work_roi=roi,
                           diagnostics=diagnostics)

    def _count(self, color, evidence, *, color_order, depth_scale_m, arm_mask,
               exclusion_mask, work_roi, diagnostics):
        p = self.parameters
        cv = _opencv()
        rgb = np.asarray(color) if color_order == 'RGB' else np.asarray(color)[..., ::-1]
        shape = rgb.shape[:2]
        roi = np.ones(shape, bool) if work_roi is None else bool_mask(work_roi, shape, 'work_roi').copy()
        excluded = np.zeros(shape, bool)
        for name, value in (('arm_mask', arm_mask), ('exclusion_mask', exclusion_mask)):
            if value is not None:
                excluded |= bool_mask(value, shape, name)
        delta = evidence.depth_difference_m
        high = np.zeros(shape, bool)
        if p['max_reference_separation_m'] is not None:
            high = evidence.depth_valid_mask & (delta > p['max_reference_separation_m'])
        excluded = _dilate(excluded | high, p['exclusion_guard_px']) & roi
        domain = roi & ~excluded
        usable_color = evidence.color_usable_mask & domain
        observed = usable_color & evidence.depth_valid_mask
        candidate = evidence.object_mask & domain
        features_image = _chromaticity(rgb)
        maximum = rgb.max(axis=2).astype(float)
        saturation = (maximum - rgb.min(axis=2)) / np.maximum(maximum, 1.)
        dark = candidate & (maximum < p['min_brightness'])
        # Do not let missing-depth occluders bypass a declared upper band.
        unbounded = (candidate & ~evidence.depth_valid_mask
                     if p['max_reference_separation_m'] is not None else np.zeros(shape, bool))
        eligible = candidate & usable_color & ~dark & ~unbounded
        features = features_image[eligible]
        centers, overflow = _palette(features, p['color_tolerance'], p['max_color_groups']) if len(features) else (np.empty((0, 3)), False)
        labels_flat, ambiguous_flat = _assign(features, centers, p['color_tolerance'], p['color_margin'])
        labels = np.zeros(shape, np.int32)
        labels[eligible] = labels_flat
        appearance_ambiguous = np.zeros(shape, bool)
        appearance_ambiguous[eligible] = ambiguous_flat
        groups = []
        output = np.zeros(shape, np.int32)
        rejected = candidate & ~eligible
        unresolved = candidate & ~eligible
        reasons = []
        boundary = np.zeros(shape, bool)
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
        # Nonrectangular ROI boundaries are crop boundaries too.
        boundary |= roi & _dilate(~roi, 1)
        for label in range(1, len(centers)+1):
            mask = labels == label
            n, fragments, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            areas = stats[1:, cv.CC_STAT_AREA]
            retained = np.zeros(n, bool)
            retained[1:] = areas >= p['min_fragment_area']
            part = retained[fragments]
            rejected |= mask & ~part
            if int(part.sum()) < p['min_object_area']:
                rejected |= part
                continue
            group_reasons = []
            support = part & evidence.depth_foreground_mask & ~evidence.depth_threshold_margin_mask
            depth_fraction = float(support.sum() / part.sum())
            if depth_fraction < p['min_depth_support_fraction']:
                group_reasons.append('insufficient_depth_support_shadow_or_thin_object')
                unresolved |= part
            # Check EACH fragment: a distant shadow with the same color must not
            # borrow depth support from a real object elsewhere in the frame.
            supported_counts = np.bincount(fragments[support], minlength=n)
            fragment_fraction = supported_counts / np.maximum(stats[:, cv.CC_STAT_AREA], 1)
            unsupported = retained & (fragment_fraction < p['min_depth_support_fraction'])
            unsupported_fragment_pixels = int(stats[unsupported, cv.CC_STAT_AREA].sum())
            unresolved |= unsupported[fragments]
            if unsupported_fragment_pixels and 'insufficient_depth_support_shadow_or_thin_object' not in group_reasons:
                group_reasons.append('fragment_without_depth_support')
            if float(np.median(saturation[part])) < p['min_saturation']:
                group_reasons.append('achromatic_brightness_identity_ambiguous')
                unresolved |= part
            pixels = features_image[part]
            center = pixels[0] + np.mean(pixels - pixels[0], axis=0)
            spread = float(np.quantile(np.linalg.norm(pixels-center, axis=1), .95))
            if spread > p['color_tolerance'] * .8:
                group_reasons.append('broad_color_distribution')
                unresolved |= part
            group_id = len(groups) + 1
            output[part] = group_id
            yy, xx = np.nonzero(part)
            groups.append(dict(id=group_id, area_px=int(part.sum()),
                fragment_count=int(retained.sum()), fragment_areas_px=sorted(map(int, areas[retained[1:]]), reverse=True),
                representative_rgb=np.rint(np.median(rgb[part], axis=0)).astype(int).tolist(),
                chromaticity=center.tolist(), color_spread_p95=spread,
                depth_supported_fraction=depth_fraction, unsupported_fragment_pixels=unsupported_fragment_pixels,
                bbox_xyxy=[int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1],
                touches_crop=bool((part & boundary).any()), reasons=group_reasons))
        # Features not admitted to any center, margins, and area filtering are
        # visible in masks. Large unresolved residue prevents an accepted count.
        rejected |= candidate & (output == 0)
        unresolved |= eligible & (labels == 0)
        rejected_large = False
        n, _, stats, _ = cv.connectedComponentsWithStats(rejected.astype(np.uint8), connectivity=8)
        if n > 1 and np.any(stats[1:, cv.CC_STAT_AREA] >= p['min_object_area']):
            rejected_large = True
            reasons.append('substantial_unassigned_foreground')
        if overflow:
            reasons.append('color_group_budget_exceeded')
        if appearance_ambiguous.sum() >= p['min_fragment_area']:
            reasons.append('color_assignments_ambiguous')
        if dark.sum() >= p['min_fragment_area']:
            reasons.append('foreground_too_dark_for_color_identity')
        if unbounded.any():
            reasons.append('missing_depth_cannot_apply_upper_band')
        for group in groups:
            reasons.extend(group['reasons'])
        # Refit at nearby tolerances to expose unstable counts, not a statistical
        # confidence interval or a guarantee about every possible threshold.
        stability_counts = []
        if len(features):
            for factor in (1-p['stability_fraction'], 1+p['stability_fraction']):
                tolerance = p['color_tolerance'] * factor
                alt_centers, alt_overflow = _palette(features, tolerance, p['max_color_groups'])
                alt_labels, _ = _assign(features, alt_centers, tolerance, p['color_margin'])
                alt_image = np.zeros(shape, np.int32)
                alt_image[eligible] = alt_labels
                supported_groups = 0
                for label in range(1, len(alt_centers)+1):
                    _, _, stats, _ = cv.connectedComponentsWithStats((alt_image == label).astype(np.uint8), connectivity=8)
                    areas = stats[1:, cv.CC_STAT_AREA]
                    if int(areas[areas >= p['min_fragment_area']].sum()) >= p['min_object_area']:
                        supported_groups += 1
                stability_counts.append(supported_groups)
                if alt_overflow:
                    reasons.append('color_group_budget_exceeded')
            if any(count != len(groups) for count in stability_counts):
                reasons.append('count_changes_with_color_tolerance')
        # Close centers can encode shading or genuinely distinct objects. Do not
        # quietly certify a brittle split at the nominal radius.
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if np.linalg.norm(np.array(groups[i]['chromaticity'])-groups[j]['chromaticity']) < 2*p['color_tolerance']:
                    reasons.append('nearby_color_groups_may_be_shading')
        missing = domain & ~observed
        if missing.any():
            reasons.append('incomplete_rgbd_observation')
        if (evidence.conflict_mask & domain).any():
            reasons.append('color_depth_conflict')
        background_unresolved = evidence.unresolved_mask & domain
        if background_unresolved.any():
            reasons.append('background_fusion_has_unresolved_changes')
        background_filtered = evidence.filtered_mask & domain
        if background_filtered.sum() >= p['min_object_area']:
            reasons.append('background_segmentation_filtered_substantial_change')
        reference_status = evidence.color_reference_status
        if reference_status in ('rejected', 'insufficient'):
            reasons.append('background_color_reference_unusable')
        if not domain.any():
            reasons.append('no_searchable_pixels')
        input_uncertain = (np.zeros(shape, bool) if evidence.input_uncertain_mask is None
                           else evidence.input_uncertain_mask & domain)
        cleanup_changed = (np.zeros(shape, bool) if evidence.cleanup_changed_mask is None
                           else evidence.cleanup_changed_mask & domain)
        if input_uncertain.any():
            reasons.append('external_mask_has_uncertain_pixels')
        if cleanup_changed.any():
            reasons.append('cleanup_changed_pixels_are_inferred_not_observed')
        if evidence.parameters.get('depth_evidence') == 'unavailable':
            reasons.append('external_mask_has_no_depth_evidence')
        reasons = list(dict.fromkeys(reasons))
        uncertain = (evidence.uncertain_mask | unresolved | appearance_ambiguous | rejected | missing) & domain
        count = None if reasons or rejected_large else len(groups)
        state = ('insufficient' if not domain.any() or not observed.any() else 'ambiguous' if count is None
                 else 'no_visible_candidates' if count == 0 else 'estimate')
        warnings = ['distinct_near_solid_colors_and_fixed_registered_view_are_caller_assumptions',
                    'color_groups_are_not_verified_physical_instances_or_count_bounds',
                    'same_color_fragments_are_merged_as_an_identity_hypothesis',
                    'fully_hidden_outside_crop_and_below_area_threshold_objects_are_not_counted',
                    'proportional_encoded_rgb_shading_model_is_not_general_illumination_invariance',
                    'brightness_only_and_arbitrarily_close_colors_cannot_be_reliably_separated',
                    'arm_masks_depth_bands_and_area_filters_are_not_semantic_arm_detection',
                    'depth_separation_is_camera_z_not_table_normal_height',
                    'synthetic_tests_do_not_establish_real_image_accuracy']
        if evidence.parameters['input_mode'] == 'external_mask':
            warnings.append('external_mask_selection_and_coverage_are_caller_assumptions')
            if cleanup_changed.any():
                warnings.append('cleanup_changes_do_not_create_sensor_evidence')
        if rejected.any():
            warnings.append('small_or_unclassified_foreground_was_filtered_inspect_rejected_mask')
        if excluded.any():
            warnings.append('excluded_pixels_are_unobserved_not_known_empty')
        metrics = dict(search_pixels=int(domain.sum()), observed_pixels=int(observed.sum()),
                       candidate_pixels=int(candidate.sum()), grouped_pixels=int((output > 0).sum()),
                       rejected_pixels=int(rejected.sum()), excluded_pixels=int(excluded.sum()),
                       high_separation_pixels=int(high.sum()), uncertain_pixels=int(uncertain.sum()),
                       missing_rgbd_pixels=int(missing.sum()), background_unresolved_pixels=int(background_unresolved.sum()),
                       background_filtered_pixels=int(background_filtered.sum()), tolerance_candidate_counts=stability_counts,
                       color_reference_status=reference_status)
        if evidence.parameters['input_mode'] == 'external_mask':
            metrics.update(input_uncertain_pixels=int(input_uncertain.sum()),
                           cleanup_changed_pixels=int(cleanup_changed.sum()))
        debug = {}
        if diagnostics:
            debug = dict(depth_difference_m=delta.copy(), chromaticity=features_image,
                         depth_foreground_mask=evidence.depth_foreground_mask & domain,
                         high_separation_mask=high, appearance_ambiguous_mask=appearance_ambiguous,
                         background_unresolved_mask=background_unresolved, background_filtered_mask=background_filtered,
                         dark_foreground_mask=dark, unbounded_foreground_mask=unbounded,
                         background=evidence.summary)
            if evidence.color_foreground_mask is not None:
                debug['color_foreground_mask'] = evidence.color_foreground_mask & domain
            if evidence.parameters['input_mode'] == 'external_mask':
                debug.update(input_uncertain_mask=input_uncertain.copy(),
                             cleanup_changed_mask=cleanup_changed.copy())
        parameters = dict(p, **evidence.parameters,
                          color_order=color_order, depth_scale_m=None if depth_scale_m is None else float(depth_scale_m),
                          registration='caller_declared' if depth_scale_m is not None else 'shared_color_mask_grid',
                          color_feature='255 * RGB / sum(RGB)',
                          histogram_bin_width=min(4., p['color_tolerance']/4.) if p['color_tolerance'] >= 1. else 0.,
                          connectivity=8)
        return ObjectCountResult(count, len(groups), state, groups, reasons, warnings, metrics, parameters,
                                 output, output > 0, candidate, uncertain, excluded, rejected, observed, debug)
