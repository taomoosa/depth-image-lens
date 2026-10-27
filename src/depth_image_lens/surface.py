"""Observed elevated/nonflat depth evidence; never a single-view layer counter."""
from dataclasses import dataclass, field
import numpy as np
from .debug import DebugResultMixin
from .depth import _numeric_image, _validate_depth_scale
from .geometry import bool_mask, CameraIntrinsics, TablePlane, mask_observation_regions
from .color import _erode


@dataclass(frozen=True)
class SurfaceResult(DebugResultMixin):
    state: str
    elevation_state: str
    shape_state: str
    metrics: dict
    regions: list
    parameters: dict
    object_mask: np.ndarray = field(repr=False)
    elevated_mask: np.ndarray = field(repr=False)
    nonflat_mask: np.ndarray = field(repr=False)
    unknown_mask: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict,repr=False)
    layer_status: str = 'unknown'
    scope: str = 'visible_surface_evidence_not_multilayer_proof'
    def to_dict(self):
        return {k:v for k,v in vars(self).items() if not isinstance(v,np.ndarray) and k!='diagnostics'}


class DepthSurfaceDetector:
    """Measure height/reference separation and local shape on a selected object.

    A sloping plane can be planar yet elevated. A single wrinkled/draped sheet
    can be nonflat. Both are reported without asserting stacked material.
    """
    def __init__(self, *, cell_px=24, min_points=20, min_valid_fraction=.7,
                 elevation_m=.025, reference_error_m=.005, residual_m=.005,
                 step_m=.015, min_step_pairs=3, boundary_guard_px=1):
        if any(type(v) is not int or v<1 for v in (cell_px,min_points,min_step_pairs)) or cell_px<4:
            raise ValueError('positive integer sizes; cell_px >=4 required')
        if type(boundary_guard_px) is not int or not 0<=boundary_guard_px<=3:raise ValueError('boundary_guard_px integer 0..3 required')
        if not np.isfinite([min_valid_fraction,elevation_m,reference_error_m,residual_m,step_m]).all() or not 0<min_valid_fraction<=1 or min(elevation_m,residual_m,step_m)<=0 or reference_error_m<0:
            raise ValueError('invalid finite thresholds')
        self.parameters=dict(cell_px=cell_px,min_points=min_points,min_valid_fraction=float(min_valid_fraction),
            elevation_m=float(elevation_m),reference_error_m=float(reference_error_m),residual_m=float(residual_m),
            step_m=float(step_m),min_step_pairs=min_step_pairs,boundary_guard_px=boundary_guard_px)

    def detect(self, depth, object_mask, *, depth_scale_m=.001, intrinsics=None, table=None,
               reference_depth=None, observed_mask=None, reference_valid=None, exclude_mask=None,
               diagnostics=False):
        """HxW optical-axis Z, explicit metres/count for all numeric dtypes.

        Table+intrinsics measure normal height; reference_depth is a same-pixel
        baseline surface using the same scale (Z separation, not thickness).
        Neither reference is required for independent local shape measurements.
        """
        _validate_depth_scale(depth_scale_m)
        if type(diagnostics) is not bool:raise ValueError('diagnostics must be bool')
        if reference_valid is not None and reference_depth is None:raise ValueError('reference_valid requires reference_depth')
        if table is not None and reference_depth is not None:raise ValueError('choose table or reference_depth')
        if intrinsics is not None and not isinstance(intrinsics,CameraIntrinsics):raise ValueError('CameraIntrinsics required')
        if table is not None and (not isinstance(table,TablePlane) or intrinsics is None):raise ValueError('table requires TablePlane and CameraIntrinsics')
        z=_numeric_image(depth,'depth')*depth_scale_m;shape=z.shape
        obj=bool_mask(object_mask,shape,'object_mask').copy()
        if exclude_mask is not None:obj &= ~bool_mask(exclude_mask,shape,'exclude_mask')
        obs=np.ones(shape,bool) if observed_mask is None else bool_mask(observed_mask,shape,'observed_mask')
        valid=np.isfinite(z)&(z>0)&obs;interior=_erode(obj,self.parameters['boundary_guard_px'])
        points=None
        if intrinsics is not None:
            with np.errstate(over='ignore',invalid='ignore'):points=intrinsics.deproject(np.where(valid,z,0))
            valid &= np.isfinite(points).all(axis=2)
        height=np.full(shape,np.nan);refkind='missing'
        if table is not None:
            refkind='table_normal_height_m';height=table.height(points)
        elif reference_depth is not None:
            ref=_numeric_image(reference_depth,'reference_depth')*depth_scale_m
            if ref.shape!=shape:raise ValueError('reference/depth shape mismatch')
            rv=np.isfinite(ref)&(ref>0)
            if reference_valid is not None:rv &= bool_mask(reference_valid,shape,'reference_valid')
            with np.errstate(invalid='ignore'):height=np.where(rv,ref-z,np.nan)
            refkind='reference_camera_z_separation_m'
        elif reference_valid is not None:raise ValueError('reference_valid requires reference_depth')
        p=self.parameters;measured=obj&valid;hv=measured&np.isfinite(height)
        elevated=hv&(height-p['reference_error_m']>=p['elevation_m'])
        nonflat=np.zeros(shape,bool);computed=np.zeros(shape,bool);regions=[]
        component_labels,_=mask_observation_regions(obj,valid)
        # Regions are fixed inspection cells, not semantic fold boundaries.
        for y0 in range(0,shape[0],p['cell_px']):
            for x0 in range(0,shape[1],p['cell_px']):
                sl=np.s_[y0:min(y0+p['cell_px'],shape[0]),x0:min(x0+p['cell_px'],shape[1])]
                selected=interior[sl]
                if not selected.any():continue
                for component_id in np.unique(component_labels[sl][selected]):
                    component_selection=selected&(component_labels[sl]==component_id)
                    ok=component_selection&valid[sl];n=int(ok.sum());coverage=n/int(component_selection.sum())
                    item=dict(id=len(regions)+1,component_id=int(component_id),box_xyxy=[x0,y0,sl[1].stop,sl[0].stop],
                        selected_pixels=int(component_selection.sum()),valid_pixels=n,valid_fraction=coverage,
                        shape_status='unknown',reasons=[],residual_p95_m=None,step_pairs=None)
                    if n<p['min_points'] or coverage<p['min_valid_fraction']:
                        item['reasons'].append('insufficient_local_observation');regions.append(item);continue
                    yy,xx=np.indices(ok.shape);design=np.c_[xx[ok],yy[ok],np.ones(n)]
                    try:
                        fit,_,rank,_=np.linalg.lstsq(design,z[sl][ok],rcond=None)
                        if rank<3:raise ValueError('degenerate_image_support')
                        trend=fit[0]*xx+fit[1]*yy+fit[2];signed=z[sl]-trend
                        if points is not None:
                            cloud=points[sl][ok];center=cloud.mean(axis=0)
                            _,sv,axes=np.linalg.svd(cloud-center,full_matrices=False)
                            if len(sv)<3 or sv[1]<1e-10:raise ValueError('degenerate_3d_support')
                            normal=axes[-1]
                            residual=np.abs((cloud-center)@normal)
                            # Optical-axis Z of a perspective plane is not affine
                            # in image coordinates. Reproject the fitted 3D plane
                            # onto each camera ray before measuring Z steps.
                            rays=np.stack(((xx+x0-intrinsics.cx)/intrinsics.fx,
                                (yy+y0-intrinsics.cy)/intrinsics.fy,np.ones_like(xx)),axis=-1)
                            denominator=rays@normal
                            if np.any(np.abs(denominator[ok])<1e-8):
                                raise ValueError('local_plane_ray_near_parallel')
                            with np.errstate(divide='ignore',invalid='ignore'):
                                predicted_z=float(normal@center)/denominator
                            if not np.isfinite(predicted_z[ok]).all() or np.any(predicted_z[ok]<=0):
                                raise ValueError('local_plane_ray_intersection_unusable')
                            signed=z[sl]-predicted_z
                        else:residual=np.abs(signed[ok])
                        p95=float(np.quantile(residual,.95))
                        if not np.isfinite(p95):raise ValueError('nonfinite_fit')
                        pairs=0
                        for dy,dx in ((0,1),(1,0)):
                            a=np.s_[:ok.shape[0]-dy or None,:ok.shape[1]-dx or None]
                            b=np.s_[dy:,dx:]
                            with np.errstate(invalid='ignore'):
                                pairs+=int(((ok[a]&ok[b])&(np.abs(signed[a]-signed[b])>p['step_m'])).sum())
                        bad=p95>p['residual_m'] or pairs>=p['min_step_pairs']
                        computed[sl]|=ok
                        if bad:nonflat[sl]|=ok
                        item.update(shape_status='nonflat_candidate' if bad else 'locally_planar',
                            residual_p95_m=p95,step_pairs=pairs)
                        if p95>p['residual_m']:item['reasons'].append('local_plane_residual')
                        if pairs>=p['min_step_pairs']:item['reasons'].append('internal_depth_steps')
                    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                        item['reasons'].append(str(exc))
                    regions.append(item)
        coverage=float(measured.sum()/max(obj.sum(),1));height_coverage=float(hv.sum()/max(obj.sum(),1))
        _,high_regions=mask_observation_regions(elevated,elevated)
        high_sufficient=any(region['pixels']>=p['min_points'] for region in high_regions)
        shape_sufficient=int(nonflat.sum())>=p['min_points']
        elevation_state='observed_elevation' if high_sufficient else 'no_observed_elevation' if hv.sum()>=p['min_points'] and np.array_equal(hv,obj) else 'unknown'
        shape_state='observed_nonflat' if shape_sufficient else 'no_observed_nonflat' if computed.sum()>=p['min_points'] and np.array_equal(computed,interior) else 'unknown'
        state='anomaly_candidate' if high_sufficient or shape_sufficient else 'no_observed_anomaly' if elevation_state!='unknown' and shape_state!='unknown' else 'unknown'
        metrics=dict(reference=refkind,object_pixels=int(obj.sum()),valid_pixels=int(measured.sum()),valid_fraction=coverage,
            height_valid_fraction=height_coverage,height_quantiles_m=[float(x) for x in np.quantile(height[hv],[.1,.5,.95])] if hv.any() else None,
            elevated_pixels=int(elevated.sum()),elevated_regions=high_regions,nonflat_pixels=int(nonflat.sum()),computed_shape_pixels=int(computed.sum()),
            shape_geometry='camera_3d_local_plane' if points is not None else 'affine_image_z_approximation',
            step_geometry='plane_ray_z_residual' if points is not None else 'affine_image_z_residual',
            warnings=['height_is_not_layer_count','single_draped_sheet_or_air_gap_can_mimic_pile',
                'cell_masks_are_not_precise_fold_boundaries','fixed_grid_and_pixel_thresholds_depend_on_resolution'])
        debug=dict(valid_mask=measured,shape_computed_mask=computed,interior_mask=interior) if diagnostics else {}
        return SurfaceResult(state,elevation_state,shape_state,metrics,regions,dict(p),obj,elevated,nonflat,obj&~computed,debug)
