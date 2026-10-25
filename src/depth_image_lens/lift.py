"""Static visible-region elevation, independent of grasp evidence."""
from dataclasses import dataclass, asdict, field
import numpy as np
from .geometry import CameraIntrinsics, TablePlane, bool_mask, largest_component


@dataclass(frozen=True)
class LiftResult:
    state: str
    reason: str
    scope: str = "visible_region"
    whole_object_lifted: str = "unknown"
    valid_fraction: float = 0.
    elevated_fraction: float | None = None
    near_table_fraction: float | None = None
    height_quantiles_m: tuple | None = None
    largest_elevated_component_pixels: int = 0
    validation_status: str = "not_validated"
    stage_results: dict = field(default_factory=dict)


    def to_dict(self):
        """Return JSON-safe scalar results; no source paths or image arrays."""
        return asdict(self)


class LiftDetector:
    def __init__(self, *, lift_height_m=.03, near_table_m=.01, height_error_m=.005,
                 min_pixels=20, min_component_pixels=12, min_lift_fraction=.1,
                 min_valid_fraction=.7):
        """Thresholds require calibration; height_error_m is a caller assumption.

        It must cover depth AND plane errors in signed-height units. No
        automatic sensor-noise model or guarantee is supplied.
        """
        values = [lift_height_m, near_table_m, height_error_m, min_lift_fraction, min_valid_fraction]
        if (not np.isfinite(values).all() or not 0 <= near_table_m < lift_height_m
                or height_error_m < 0 or not 0 < min_lift_fraction <= 1
                or not 0 < min_valid_fraction <= 1
                or any(type(x) is not int or x < 1 for x in (min_pixels,min_component_pixels))):
            raise ValueError("invalid lift settings")
        self.lift_height_m = lift_height_m; self.near_table_m = near_table_m
        self.height_error_m = height_error_m; self.min_pixels = min_pixels
        self.min_component_pixels = min_component_pixels
        self.min_lift_fraction = min_lift_fraction; self.min_valid_fraction = min_valid_fraction

    def detect_16uc1(self, depth, intrinsics, table, object_mask, *, depth_scale_m=.001, **options):
        """Read uint16 optical-axis depth counts; default 1 mm/count is explicit.

        No difference image or hand ROI is needed. A positive-difference mask
        may help select candidates, but cannot establish object identity/height.
        """
        from .depth import _depth_16uc1_to_meters
        return self.detect(_depth_16uc1_to_meters(depth, depth_scale_m=depth_scale_m),
                           intrinsics, table, object_mask, **options)

    def detect(self, depth_m, intrinsics: CameraIntrinsics, table: TablePlane,
               object_mask, *, exclude_mask=None, observed_mask=None, whole_object_visible=False):
        """All masks share the rectified depth grid; missing depth is not empty space.

        object_mask identifies the visible object, not the table's projected
        ROI. A raised object can project outside that ROI. Supply an exclusion
        mask for robot/background contamination. Whole-object visibility must
        include self-occluded parts, not merely staying inside the image.
        """
        z = np.asarray(depth_m, float)
        if z.ndim != 2:
            raise ValueError("HxW metric optical-axis depth required")
        obj = bool_mask(object_mask,z.shape,"object_mask")
        exc = np.zeros(z.shape,bool) if exclude_mask is None else bool_mask(exclude_mask,z.shape,"exclude_mask")
        obs = np.ones(z.shape,bool) if observed_mask is None else bool_mask(observed_mask,z.shape,"observed_mask")
        if type(whole_object_visible) is not bool:
            raise ValueError("whole_object_visible must be boolean")
        eligible = obj & ~exc
        valid = eligible & obs & np.isfinite(z) & (z > 0)
        coverage = float(valid.sum()/max(eligible.sum(),1))
        sufficient=valid.sum()>=self.min_pixels and coverage>=self.min_valid_fraction
        warnings=[] if sufficient else ['limited_observed_object_support']
        if not valid.any():
            return LiftResult("unknown","insufficient_observed_object",valid_fraction=coverage,
                stage_results={name:{'status':'insufficient_input','reason':'no_valid_object_depth'}
                    for name in ('height','visible_lift','whole_object')})
        with np.errstate(over="ignore", invalid="ignore"):
            h = table.height(intrinsics.deproject(np.where(valid,z,0)))
        finite=valid&np.isfinite(h)
        if not finite.any():
            return LiftResult("unknown", "nonfinite_height_geometry", valid_fraction=coverage,
                stage_results={name:{'status':'insufficient_input','reason':'no_finite_deprojection'}
                    for name in ('height','visible_lift','whole_object')})
        if not np.array_equal(finite,valid):warnings.append('some_nonfinite_deprojection')
        valid=finite
        sufficient &= valid.sum()>=self.min_pixels and valid.sum()/max(eligible.sum(),1)>=self.min_valid_fraction
        if not sufficient and 'limited_observed_object_support' not in warnings:
            warnings.append('limited_observed_object_support')
        high = valid & (h-self.height_error_m > self.lift_height_m)
        near = valid & (np.abs(h)+self.height_error_m <= self.near_table_m+1e-12)
        component = largest_component(high)
        fraction = float(high.sum()/valid.sum()); near_fraction = float(near.sum()/valid.sum())
        state = "lifted" if component >= self.min_component_pixels and fraction >= self.min_lift_fraction else "not_lifted" if near_fraction >= .8 else "unknown"
        if not sufficient: state="unknown"
        whole = "unknown"
        if largest_component(near) >= self.min_component_pixels:
            whole = "no"  # near-table points do not meet the lift-height criterion
        elif (sufficient and component>=self.min_component_pixels and whole_object_visible
              and np.array_equal(obj,valid) and np.all(high[obj])):
            whole = "yes"
        whole_warnings=list(warnings)
        if not whole_object_visible:whole_warnings.append('whole_object_visibility_unverified')
        if not np.array_equal(obj,valid):whole_warnings.append('whole_object_not_fully_observed')
        if component<self.min_component_pixels:whole_warnings.append('insufficient_elevated_component')
        return LiftResult(state,"nonfinite_height_geometry" if "some_nonfinite_deprojection" in warnings else "height_geometry" if sufficient else "insufficient_observed_object",whole_object_lifted=whole,
                          valid_fraction=coverage,elevated_fraction=fraction,
                          near_table_fraction=near_fraction,
                          height_quantiles_m=tuple(float(x) for x in np.quantile(h[valid],[.1,.5,.9])),
                          largest_elevated_component_pixels=component,
                          stage_results={'height':{'status':'computed','points':int(valid.sum()),'warnings':warnings},
                            'visible_lift':{'status':'computed','value':state,'warnings':warnings},
                            'whole_object':{'status':'computed' if whole!='unknown' else 'insufficient_input','value':whole,
                                'warnings':whole_warnings,'finite_points':int(valid.sum()),'min_pixels':self.min_pixels,
                                'elevated_component_pixels':component,'min_component_pixels':self.min_component_pixels}})
