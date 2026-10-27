"""Small SVD fit for a caller-selected empty-table ROI; no segmentation."""
from dataclasses import dataclass, asdict, field
import numpy as np
from .debug import DebugResultMixin
from .geometry import TablePlane, bool_mask
from .depth import _numeric_image, _validate_depth_scale


@dataclass(frozen=True)
class PlaneFitResult(DebugResultMixin):
    plane: TablePlane
    point_count: int
    valid_fraction: float
    residual_median_m: float
    residual_p95_m: float
    residual_rmse_m: float
    in_plane_rms_spread_m: tuple[float, float]
    orientation: str
    warnings: tuple[str, ...] = ()

    diagnostics: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        """Strict-JSON-safe summary, with no paths or image arrays."""
        return {k:v for k,v in asdict(self).items() if k!='diagnostics'}


def fit_table_plane(depth, intrinsics, table_mask, *, depth_scale_m=.001,
                    positive_direction=None, min_points=20, min_spread_m=.01,
                    residual_warning_m=.005, diagnostics=False):
    """Fit optical-axis depth in the camera frame using valid ROI points.

    Scale is metres per input count for EVERY dtype: .001 for millimetres,
    1.0 for metres. Default supports uint16 16UC1 without dtype inference.
    Positive side faces the camera origin unless positive_direction is given.
    This least-squares fit is not robust to clutter or a wrong table ROI.
    """
    if type(diagnostics) is not bool:raise ValueError('diagnostics must be bool')
    _validate_depth_scale(depth_scale_m)
    if (type(min_points) is not int or min_points < 3
            or not np.isfinite([min_spread_m,residual_warning_m]).all()
            or min_spread_m <= 0 or residual_warning_m <= 0):
        raise ValueError('positive finite spread/residual settings and min_points >= 3 required')
    raw = _numeric_image(depth,'depth')
    roi = bool_mask(table_mask,raw.shape,'table_mask')
    with np.errstate(over='ignore',invalid='ignore'):
        z = raw if depth_scale_m == 1 else raw*depth_scale_m
    valid = roi & np.isfinite(z) & (z>0)
    count = int(valid.sum())
    if count < min_points:
        raise ValueError('insufficient valid table points')
    with np.errstate(over='ignore',invalid='ignore'):
        points = intrinsics.deproject(np.where(valid,z,0))[valid]
    if not np.isfinite(points).all():
        raise ValueError('nonfinite deprojected table geometry')
    center = points.mean(axis=0)
    centered = points-center
    if not np.isfinite(centered).all():
        raise ValueError('nonfinite centered table geometry')
    try:
        _,singular,axes = np.linalg.svd(centered,full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise ValueError('table SVD did not converge') from error
    spread = singular[:2]/np.sqrt(count)
    if not np.isfinite(singular).all() or spread[1] < min_spread_m:
        raise ValueError('table ROI is collinear, too narrow or degenerate')
    if singular[1]-singular[2] <= max(1e-12, singular[0]*1e-8):
        raise ValueError('table normal is not uniquely supported')
    normal = axes[2]
    direction = -center if positive_direction is None else np.asarray(positive_direction,float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('positive_direction must be a finite camera-frame vector')
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('positive direction is degenerate')
    alignment = float(normal@(direction/length))
    if abs(alignment) < 1e-6:
        raise ValueError('positive direction is parallel to the fitted plane')
    if alignment < 0: normal = -normal
    plane = TablePlane(tuple(normal),-float(normal@center))
    residual = np.abs(centered@normal)
    median,p95 = np.quantile(residual,[.5,.95])
    rmse = np.linalg.norm(residual)/np.sqrt(count)
    if not np.isfinite([median,p95,rmse]).all():
        raise ValueError('nonfinite fit diagnostics')
    warnings = ('large_plane_residual',) if p95 > residual_warning_m else ()
    return PlaneFitResult(plane,count,float(count/max(int(roi.sum()),1)),
                          float(median),float(p95),float(rmse),tuple(float(v) for v in spread),
                          'toward_camera_origin' if positive_direction is None else 'caller_direction',warnings,
        dict(table_mask=roi.copy(),valid_mask=valid,parameters=dict(min_points=min_points,
            min_spread_m=min_spread_m,residual_warning_m=residual_warning_m)) if diagnostics else {})
