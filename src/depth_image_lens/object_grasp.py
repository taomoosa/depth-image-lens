"""Object-only static support inference under an explicit suspension rule."""
from dataclasses import dataclass, field
import numpy as np
from .depth import _numeric_image, _depth_16uc1_to_meters, depth_pair_16uc1_to_meters
from .geometry import CameraIntrinsics, TablePlane, bool_mask


@dataclass(frozen=True)
class ObjectOnlyGraspResult:
    held: str
    bilateral: str
    reason: str
    bilateral_reason: str
    metrics: dict
    thresholds: dict
    warnings: list
    scope: str = 'visible_selected_region'
    status: str = 'rule_based_inference'
    physical_grasp: str = 'unknown'
    validation_status: str = 'not_validated'
    diagnostics: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if k != 'diagnostics'}


class ObjectOnlyGraspDetector:
    """Infer held from elevation, and bilateral support from sustained visible width.

    Caller selects only the deformable object, assumes static suspension by hands,
    and provides a meaningful table plane or same-pixel empty-background difference.
    No hand masks, physical identity, learned segmentation or hidden geometry.
    """
    def __init__(self, *, min_lift_m=.03, min_width_px=40, min_pixels=20,
                 min_width_rows=3, min_lift_fraction=.1, min_valid_fraction=.7,
                 reference_error_m=.005, depth_jump_m=.1):
        for name, value in [('min_width_px', min_width_px), ('min_pixels', min_pixels),
                            ('min_width_rows', min_width_rows)]:
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        values = [min_lift_m, min_lift_fraction, min_valid_fraction, reference_error_m, depth_jump_m]
        if (not np.isfinite(values).all() or min_lift_m <= 0 or reference_error_m < 0
                or depth_jump_m <= 0 or not 0 < min_lift_fraction <= 1
                or not 0 < min_valid_fraction <= 1):
            raise ValueError('finite positive thresholds/fractions required; error must be nonnegative')
        self.thresholds = dict(min_lift_m=float(min_lift_m), min_width_px=min_width_px,
            min_pixels=min_pixels, min_width_rows=min_width_rows,
            min_lift_fraction=float(min_lift_fraction), min_valid_fraction=float(min_valid_fraction),
            reference_error_m=float(reference_error_m), depth_jump_m=float(depth_jump_m))

    def detect_16uc1(self, depth, object_mask, *, difference=None, depth_scale_m=.001, **options):
        """uint16 arrays; positive clipped background-current uses the same count scale."""
        if difference is None:
            z = _depth_16uc1_to_meters(depth, depth_scale_m=depth_scale_m)
            delta = None
        else:
            z, delta = depth_pair_16uc1_to_meters(depth, difference, depth_scale_m=depth_scale_m)
        return self.detect(z, object_mask, depth_unit='m', difference=delta, **options)

    def detect(self, depth, object_mask, *, depth_unit, difference=None,
               intrinsics=None, table=None, observed_mask=None, background_valid=None,
               whole_object_visible=False, mask_complete=False, diagnostics=False):
        """Explicit m/mm optical-axis Z; difference uses the same unit, if supplied.

        Difference means background-current (signed or clipped positive), never
        absolute height. Zeros may be arbitrary exclusions, so cannot prove absence.
        Exactly one reference route is allowed, or none for an unknown result.
        """
        if depth_unit not in ('m', 'mm'):
            raise ValueError('explicit m or mm depth_unit required')
        if any(type(v) is not bool for v in (whole_object_visible, mask_complete, diagnostics)):
            raise ValueError('visibility/completeness/diagnostics must be bool')
        if table is not None and difference is not None:
            raise ValueError('choose table or difference reference, not both')
        if (table is None) != (intrinsics is None):
            raise ValueError('table and intrinsics must be supplied together')
        if table is not None and (not isinstance(table, TablePlane) or not isinstance(intrinsics, CameraIntrinsics)):
            raise ValueError('TablePlane and CameraIntrinsics required')
        z = _numeric_image(depth, 'depth') * (.001 if depth_unit == 'mm' else 1.)
        obj = bool_mask(object_mask, z.shape, 'object_mask')
        obs = np.ones(z.shape, bool) if observed_mask is None else bool_mask(observed_mask, z.shape, 'observed_mask')
        bg = np.ones(z.shape, bool) if background_valid is None else bool_mask(background_valid, z.shape, 'background_valid')
        valid = np.isfinite(z) & (z > 0) & obs
        evidence = np.full(z.shape, np.nan)
        reference = 'missing'
        if table is not None:
            reference = 'table_normal_height_m'
            # Deproject only finite placeholders; never treat placeholders as observations.
            with np.errstate(over='ignore', invalid='ignore'):
                evidence = table.height(intrinsics.deproject(np.where(valid, z, 0.)))
        elif difference is not None:
            reference = 'background_camera_z_separation_m'
            evidence = _numeric_image(difference, 'difference') * (.001 if depth_unit == 'mm' else 1.)
            if evidence.shape != z.shape:
                raise ValueError('difference shape must match depth')
            valid &= bg
        valid &= np.isfinite(evidence)
        selected = obj & valid
        t = self.thresholds
        lifted = selected & (evidence - t['reference_error_m'] >= t['min_lift_m'])
        # Local four-neighbor continuity, no morphology, hole filling or component union.
        remaining = lifted.copy(); components = []; labels = np.zeros(z.shape, np.int32)
        for y, x in zip(*np.nonzero(lifted)):
            if not remaining[y, x]:
                continue
            remaining[y, x] = False; stack = [(int(y), int(x))]; points = []
            while stack:
                a, b = stack.pop(); points.append((a, b))
                for c, d in ((a-1,b), (a+1,b), (a,b-1), (a,b+1)):
                    if 0 <= c < z.shape[0] and 0 <= d < z.shape[1] and remaining[c,d]:
                        tolerance = 4*np.finfo(float).eps*max(abs(z[a,b]), abs(z[c,d]), t['depth_jump_m'])
                        if abs(z[a,b]-z[c,d]) <= t['depth_jump_m'] + tolerance:
                            remaining[c,d] = False; stack.append((c,d))
            rows = {}
            for a, b in points:
                rows.setdefault(a, []).append(b); labels[a,b] = len(components)+1
            runs = {}
            for row, xs in rows.items():
                xs.sort(); best = length = 1
                for left, right in zip(xs, xs[1:]):
                    tolerance = 4*np.finfo(float).eps*max(abs(z[row,left]), abs(z[row,right]), t['depth_jump_m'])
                    continuous = right == left+1 and abs(z[row,left]-z[row,right]) <= t['depth_jump_m']+tolerance
                    length = length+1 if continuous else 1; best = max(best, length)
                runs[row] = best
            width = max((min(runs.get(row+offset, 0) for offset in range(t['min_width_rows']))
                         for row in runs), default=0)
            components.append(dict(id=len(components)+1, pixels=len(points),
                sustained_width_px=width, max_contiguous_row_width_px=max(runs.values()),
                qualifying_width_rows=sum(v >= t['min_width_px'] for v in runs.values())))
        count = int(obj.sum()); count_valid = int(selected.sum())
        coverage = count_valid/max(count, 1); fraction = int(lifted.sum())/max(count_valid, 1)
        eligible = [c for c in components if c['pixels'] >= t['min_pixels']
                    and c['pixels']/max(count_valid, 1) >= t['min_lift_fraction']]
        wide = [c for c in eligible if c['sustained_width_px'] >= t['min_width_px']]
        cropped = bool(obj[0].any() or obj[-1].any() or obj[:,0].any() or obj[:,-1].any())
        warnings = ['assumes_static_hand_supported_object', 'single_hand_stretch_or_other_support_can_mimic_bilateral']
        if reference == 'background_camera_z_separation_m':
            warnings += ['background_separation_is_not_table_height', 'requires_task_specific_empty_background_interpretation']
        if not whole_object_visible: warnings.append('whole_object_visibility_unverified')
        if not mask_complete: warnings.append('mask_completeness_unverified')
        if cropped: warnings.append('cropped_visible_width_is_only_a_lower_bound')
        if coverage < t['min_valid_fraction']: warnings.append('reference_coverage_low')
        if len(eligible) > 1: warnings.append('multiple_components_not_combined')
        held = bilateral = 'unknown'; reason = 'insufficient_elevation_support'
        if reference == 'missing': reason = 'missing_reference'
        elif count == 0: reason = 'no_selected_object'
        elif coverage >= t['min_valid_fraction'] and eligible and fraction >= t['min_lift_fraction']:
            held = 'held'; reason = 'sufficient_elevated_connected_region'
            if wide: bilateral = 'bilateral'
        elif (reference == 'table_normal_height_m' and whole_object_visible and mask_complete
                and not cropped and count >= t['min_pixels'] and count_valid == count
                and np.all(np.abs(evidence[selected]) + t['reference_error_m'] < t['min_lift_m'])):
            held = 'not_held'; reason = 'fully_observed_region_near_table_under_elevation_rule'
            warnings.append('near_table_does_not_disprove_actual_grasp')
        chosen = max(wide or eligible, key=lambda c: c['pixels'], default=None)
        metrics = dict(reference=reference, object_pixels=count, valid_pixels=count_valid,
            valid_fraction=coverage, elevated_pixels=int(lifted.sum()), elevated_fraction=fraction,
            reference_quantiles_m=[float(v) for v in np.quantile(evidence[selected], [.1,.5,.9])] if count_valid else None,
            components=components, selected_component_id=chosen['id'] if chosen else None,
            visible_width_px=chosen['sustained_width_px'] if chosen else None,
            width_axis='image_x', width_is_lower_bound=cropped or not whole_object_visible)
        debug = dict(valid_reference_mask=selected, elevated_mask=lifted, component_labels=labels) if diagnostics else {}
        bilateral_reason = ('sufficient_same_component_visible_width' if bilateral == 'bilateral'
                            else 'insufficient_visible_width_not_unilateral' if held == 'held'
                            else 'elevated_support_not_established')
        return ObjectOnlyGraspResult(held, bilateral, reason, bilateral_reason, metrics, dict(t), warnings, diagnostics=debug)
