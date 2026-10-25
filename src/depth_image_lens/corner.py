"""Visible contour-corner candidates with local depth checks, not layer proof."""
from dataclasses import dataclass, field
import numpy as np
from .depth import _numeric_image, _validate_depth_scale
from .geometry import bool_mask, CameraIntrinsics


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise ImportError('OverheadCornerDetector needs the optional opencv-python-headless extra') from error
    return cv2


def _angle(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / denominator, -1., 1.)))) if denominator > 1e-9 else None


def _line(points):
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points-center, full_matrices=False)
    direction = axes[0]
    residual = (points-center) @ np.array([-direction[1], direction[0]])
    return center, direction, float(np.sqrt(np.mean(residual**2)))


def _sides(contour, index, distance):
    sides = []
    for step in (-1, 1):
        selected = []; traveled = 0.; previous = contour[index]
        for offset in range(1, len(contour)//2):
            point = contour[(index+step*offset) % len(contour)]
            traveled += np.linalg.norm(point-previous); previous = point
            if traveled > distance: break
            if traveled >= 2.: selected.append(point)
        sides.append(np.asarray(selected).reshape(-1, 2))
    return sides


def _fit_corner(sides, min_support=4):
    if min(map(len, sides)) < min_support: return None
    a, u, ra = _line(sides[0]); b, v, rb = _line(sides[1])
    if abs(np.linalg.det(np.stack((u, -v), axis=1))) < .1: return None
    position = a + np.linalg.solve(np.stack((u, -v), axis=1), b-a)[0]*u
    angle = _angle(a-position, b-position)
    if angle is None: return None
    return position, angle, max(ra, rb), [a.tolist(), b.tolist()]


@dataclass(frozen=True)
class CornerResult:
    candidates: list
    summary: dict
    object_mask: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        return {'candidates':self.candidates, 'summary':self.summary}


class OverheadCornerDetector:
    """Local near-right-angle contour candidates in an approximately overhead view.

    Optional OpenCV supplies contours/polygon approximation only. Independent
    NumPy line/depth fits do not identify hidden layers or verify safe grasping.
    Pixel parameters must scale with image resolution; depth thresholds are metres.
    """
    def __init__(self, *, min_area_px=100, polygon_epsilon_px=2., edge_window_px=18,
                 angle_tolerance_deg=25., line_residual_px=1., depth_radius_px=24,
                 depth_step_m=.015, depth_residual_m=.005, color_distance=35., support_policy="fixed",
                 proposal_policy="single", proposal_merge_px=3.,
                 proposal_angle_tolerance_deg=None, fitted_angle_tolerance_deg=None,
                 angle_stability_mode="reject", angle_stability_deg=10.,
                 max_intersection_shift_px=2., min_fit_support=4,
                 boundary_mixing_m=None):
        values=[min_area_px,polygon_epsilon_px,edge_window_px,angle_tolerance_deg,line_residual_px,
                depth_radius_px,depth_step_m,depth_residual_m,color_distance]
        if not np.isfinite(values).all() or min(values)<=0 or not 0<angle_tolerance_deg<60:
            raise ValueError('positive finite thresholds and angle tolerance below 60 required')
        if type(edge_window_px) is not int or edge_window_px<10 or type(depth_radius_px) is not int or depth_radius_px<5:
            raise ValueError('integer edge_window_px >= 10 and depth_radius_px >= 5 required')
        if support_policy not in ("fixed","adaptive"):
            raise ValueError("support_policy must be fixed or adaptive")
        if proposal_policy not in ("single", "multiscale"):
            raise ValueError("proposal_policy must be single or multiscale")
        if not np.isfinite(proposal_merge_px) or proposal_merge_px <= 0:
            raise ValueError("proposal_merge_px must be positive and finite")
        proposal_angle = angle_tolerance_deg if proposal_angle_tolerance_deg is None else proposal_angle_tolerance_deg
        fitted_angle = angle_tolerance_deg if fitted_angle_tolerance_deg is None else fitted_angle_tolerance_deg
        boundary_mixing = depth_step_m if boundary_mixing_m is None else boundary_mixing_m
        for value in (proposal_angle, fitted_angle):
            if not np.isfinite(value) or not 0 < value < 60:
                raise ValueError("proposal and fitted angle tolerances must be finite and between 0 and 60")
        if angle_stability_mode not in ("reject", "diagnostic"):
            raise ValueError("angle_stability_mode must be reject or diagnostic")
        for value in (angle_stability_deg, max_intersection_shift_px, boundary_mixing):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("stability, intersection shift and boundary mixing thresholds must be positive and finite")
        if type(min_fit_support) is not int or min_fit_support < 3:
            raise ValueError("min_fit_support must be an integer >= 3 per side and window")
        self.proposal_angle_tolerance_deg=float(proposal_angle)
        self.fitted_angle_tolerance_deg=float(fitted_angle)
        self.angle_stability_mode=angle_stability_mode
        self.angle_stability_deg=float(angle_stability_deg)
        self.max_intersection_shift_px=float(max_intersection_shift_px)
        self.min_fit_support=min_fit_support
        self.boundary_mixing_m=float(boundary_mixing)
        self.proposal_policy=proposal_policy; self.proposal_merge_px=float(proposal_merge_px)
        self.support_policy=support_policy
        self.min_area_px=min_area_px; self.polygon_epsilon_px=polygon_epsilon_px
        self.edge_window_px=edge_window_px; self.angle_tolerance_deg=angle_tolerance_deg
        self.line_residual_px=line_residual_px; self.depth_radius_px=depth_radius_px
        self.depth_step_m=depth_step_m; self.depth_residual_m=depth_residual_m
        self.color_distance=color_distance

    @classmethod
    def recommended_for_tuning(cls, **overrides):
        """Explicit starting preset, not robot-validated or an optimized classifier.

        Preserve the ordinary constructor defaults. Only the angle-difference
        check becomes diagnostic; both supported nondegenerate fits are required.
        """
        settings=dict(proposal_policy="multiscale", support_policy="adaptive",
                      angle_stability_mode="diagnostic")
        settings.update(overrides)
        return cls(**settings)

    def _local_fits(self, contour, index, distance):
        sides=_sides(contour,index,distance)
        short_distance=distance*.6; short_sides=_sides(contour,index,short_distance)
        adapted=False
        if (self.support_policy=='adaptive' and min(map(len,sides))>=self.min_fit_support
                and min(map(len,short_sides))<self.min_fit_support):
            proposed=_sides(contour,index,distance*.8)
            if min(map(len,proposed))>=self.min_fit_support and all(len(a)<len(b) for a,b in zip(proposed,sides)):
                short_sides=proposed;short_distance=distance*.8;adapted=True
        evidence=dict(edge_window_px=float(distance), short_edge_window_px=float(short_distance),
                      adaptive_support_used=adapted, edge_support_counts=list(map(len,sides)),
                      short_edge_support_counts=list(map(len,short_sides)))
        return _fit_corner(sides,self.min_fit_support), _fit_corner(short_sides,self.min_fit_support), evidence

    def _line_checks(self, fit, short, point, epsilon):
        """Independent upper-bound checks. None means unavailable, never a pass."""
        values={
            'line_residual_px': (fit[2] if fit is not None else None, self.line_residual_px),
            'intersection_shift_px': (float(np.linalg.norm(fit[0]-short[0])) if fit is not None and short is not None else None, self.max_intersection_shift_px),
            'angle_stability_deg': (abs(fit[1]-short[1]) if fit is not None and short is not None else None, self.angle_stability_deg),
            'fit_to_polygon_shift_px': (float(np.linalg.norm(fit[0]-point)) if fit is not None else None, max(3.,epsilon*2)),
            'fitted_angle_error_deg': (abs(fit[1]-90) if fit is not None else None, self.fitted_angle_tolerance_deg),
        }
        return {name:dict(value=value, maximum=float(limit),
                    passed=bool(value<=limit) if value is not None else None,
                    mode=self.angle_stability_mode if name=='angle_stability_deg' else 'reject')
                for name,(value,limit) in values.items()}

    def _vertices(self, source, orientation):
        """Union a bounded set of original-contour approximations, never a hull.

        Greedy radius merging compares to each retained representative, so nearby
        vertices cannot create an unbounded transitive cluster. Selection uses
        polygon geometry alone; depth and annotations do not choose proposals.
        """
        cv2=_cv2(); proposals=[]
        scales=(1.,) if self.proposal_policy=="single" else (.5,1.,2.,4.)
        for scale in scales:
            epsilon=self.polygon_epsilon_px*scale
            polygon=cv2.approxPolyDP(source,epsilon,True)[:,0,:].astype(float)
            for i,p in enumerate(polygon) if len(polygon)>=3 else []:
                before,after=polygon[i-1]-p,polygon[(i+1)%len(polygon)]-p
                angle=_angle(before,after)
                convex=(-before[0]*after[1]+before[1]*after[0])*orientation>0
                proposals.append(dict(point=p,before=before,after=after,epsilon=epsilon,
                    scales=[scale],convex=bool(convex),angle=angle))
        if self.proposal_policy=="single":return proposals
        def rank(q):
            angle=abs(q['angle']-90) if q['angle'] is not None else 180.
            extent=min(np.linalg.norm(q['before']),np.linalg.norm(q['after']))
            return (not(q['convex'] and angle<=self.proposal_angle_tolerance_deg),
                    -min(extent,2*self.edge_window_px),angle,q['epsilon'],*q['point'])
        retained=[]
        for q in sorted(proposals,key=rank):
            existing=next((r for r in retained if np.linalg.norm(q['point']-r['point'])<=self.proposal_merge_px),None)
            if existing is None:retained.append(q)
            else:existing['scales']=sorted(set(existing['scales']+q['scales']))
        return retained

    def detect(self, color, depth, *, color_order, registered_to_depth, object_mask=None,
               background_color=None, depth_scale_m=.001, valid_mask=None, intrinsics=None,
               mode='depth', diagnostics=False):
        """Aligned uint8 color and optical-axis Z, explicit metres/count for every dtype.

        Supply a bool object mask OR a matched empty-scene color image. Background
        color subtraction is a simple illumination-sensitive baseline, not semantics.
        mode='rgb' uses polygon angles; 'refined' adds line fits; 'depth' also checks
        local interior depth. Accepted always means candidate, never single layer.
        """
        cv2=_cv2()
        if registered_to_depth is not True: raise ValueError('explicit registered_to_depth=True required')
        if color_order not in ('RGB','BGR'): raise ValueError('explicit RGB or BGR required')
        if mode not in ('rgb','refined','depth'): raise ValueError('mode must be rgb, refined or depth')
        if type(diagnostics) is not bool: raise ValueError('diagnostics must be bool')
        _validate_depth_scale(depth_scale_m)
        raw=_numeric_image(depth,'depth'); rgb=np.asarray(color)
        if rgb.dtype!=np.uint8 or rgb.shape!=(*raw.shape,3): raise ValueError('aligned uint8 HxWx3 color required')
        if (object_mask is None)==(background_color is None): raise ValueError('supply exactly one object_mask or background_color')
        if intrinsics is not None and not isinstance(intrinsics,CameraIntrinsics): raise ValueError('CameraIntrinsics required')
        with np.errstate(over='ignore',invalid='ignore'): z=raw*depth_scale_m
        if np.any(np.isfinite(raw)&~np.isfinite(z)): raise ValueError('depth scaling overflow')
        valid=np.isfinite(z)&(z>0)
        if valid_mask is not None:valid &= bool_mask(valid_mask,z.shape,'valid_mask')
        if object_mask is not None:
            mask=bool_mask(object_mask,z.shape,'object_mask').copy(); provenance='caller_mask'
        else:
            bg=np.asarray(background_color)
            if bg.shape!=rgb.shape or bg.dtype!=np.uint8: raise ValueError('matching uint8 background color required')
            mask=np.linalg.norm(rgb.astype(float)-bg.astype(float),axis=2)>self.color_distance
            provenance='color_background_difference_not_semantic_segmentation'
        # Preserve depth holes for the gate, not as new holes/corners in the silhouette.
        interior=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)>2.
        outer=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)>2.
        _,component_labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        candidates=[]; debug_contours=[]; h,w=z.shape
        for contour_id,source in enumerate(contours):
            if abs(cv2.contourArea(source))<self.min_area_px:continue
            component_id=int(component_labels[source[0,0,1],source[0,0,0]])
            contour=source[:,0,:].astype(float)
            orientation=np.sign(cv2.contourArea(source,oriented=True))
            proposals=self._vertices(source,orientation)
            if diagnostics:debug_contours.append(contour.copy())
            for proposal in proposals:
                p=proposal['point'];before,after=proposal['before'],proposal['after']
                cross=(-before[0])*after[1]-(-before[1])*after[0]
                angle=_angle(before,after)
                item={'xy':p.tolist(),'polygon_xy':p.tolist(),'angle_deg':angle,'status':'accepted',
                      'reasons':[],'contour_id':contour_id,'single_layer_status':'unknown',
                      'evidence':{'convex':bool(cross*orientation>0),'stage':mode,'component_id':component_id,
                                  'contour_area_px':float(abs(cv2.contourArea(source))),
                                  'polygon_angle_deg':angle,'adjacent_edge_lengths_px':[float(np.linalg.norm(before)),float(np.linalg.norm(after))]}}
                if self.proposal_policy=='multiscale':
                    item['evidence'].update(proposal_epsilon_px=proposal['epsilon'],
                        proposal_scales=proposal['scales'],proposal_policy=self.proposal_policy)
                candidates.append(item)
                initial_angle_ok=cross*orientation>0 and angle is not None and abs(angle-90)<=self.proposal_angle_tolerance_deg
                defer_angle=self.proposal_policy=='multiscale' and mode!='rgb'
                if not initial_angle_ok and not defer_angle:
                    item.update(status='suspected',reasons=['not_convex_near_right_angle']);continue
                if min(np.linalg.norm(before),np.linalg.norm(after))<10 and not defer_angle:
                    item.update(status='unknown',reasons=['insufficient_edge_extent']);continue
                if min(p[0],p[1],w-1-p[0],h-1-p[1])<2:
                    item.update(status='unknown',reasons=['corner_at_image_boundary']);continue
                if mode!='rgb':
                    index=int(np.argmin(np.sum((contour-p)**2,axis=1)))
                    distance=min(self.edge_window_px,.45*np.linalg.norm(before),.45*np.linalg.norm(after))
                    fit,short,support=self._local_fits(contour,index,distance)
                    item['evidence'].update(support)
                    checks=self._line_checks(fit,short,p,proposal['epsilon'])
                    item['evidence']['line_checks']=checks
                    if fit is None or short is None:
                        item.update(status='unknown',reasons=['insufficient_local_line_support']);continue
                    position,angle,residual,ends=fit
                    if defer_angle:
                        a,b=np.asarray(ends)-position
                        local_convex=(-a[0]*b[1]+a[1]*b[0])*orientation>0
                        item['evidence'].update(local_convex=bool(local_convex),
                            initial_polygon_angle_passed=bool(initial_angle_ok))
                        if not local_convex:
                            item.update(status='suspected',reasons=['not_convex_local_edges']);continue
                    stability=float(np.linalg.norm(position-short[0]))
                    item['evidence'].update(line_residual_px=residual,scale_shift_px=stability,
                                            scale_angle_delta_deg=abs(angle-short[1]),edge_centers_xy=ends)
                    item.update(xy=position.tolist(),angle_deg=angle)
                    if any(check['passed'] is False and check['mode']=='reject' for check in checks.values()):
                        item.update(status='suspected',reasons=['unstable_or_curved_local_edges']);continue
                q=item['xy']
                if min(q[0],q[1],w-1-q[0],h-1-q[1])<2:
                    item.update(status='unknown',reasons=['corner_at_image_boundary']);continue
                if mode=='depth':
                    self._depth_gate(item,z,valid,component_labels,component_id,interior,outer,intrinsics)
                elif item['status']=='accepted':item['reasons']=['visible_2d_corner_only_depth_not_checked']
        # Candidate acceptance and independent measurements have different dependencies.
        # A rejected polygon angle must not erase an otherwise measurable depth patch.
        for item in candidates:
            evidence=item['evidence'];p=np.asarray(item['polygon_xy']);contour=contours[item['contour_id']][:,0,:].astype(float)
            stages={'polygon':{'status':'computed','angle_deg':evidence['polygon_angle_deg'],
                              'convex':evidence['convex']}}
            distance=min(self.edge_window_px,.45*min(evidence['adjacent_edge_lengths_px']))
            index=int(np.argmin(np.sum((contour-p)**2,axis=1)))
            fitted,short,support=self._local_fits(contour,index,distance)
            checks=self._line_checks(fitted,short,p,evidence.get('proposal_epsilon_px',self.polygon_epsilon_px))
            stages['polygon']['checks']={
                'convex': {'passed':evidence['convex']},
                'adjacent_edge_extent_px': {'value':min(evidence['adjacent_edge_lengths_px']), 'minimum':10.,
                                            'passed':min(evidence['adjacent_edge_lengths_px'])>=10},
                'angle_error_deg': {'value':abs(evidence['polygon_angle_deg']-90) if evidence['polygon_angle_deg'] is not None else None,
                                    'maximum':float(self.proposal_angle_tolerance_deg),
                                    'passed':bool(abs(evidence['polygon_angle_deg']-90)<=self.proposal_angle_tolerance_deg) if evidence['polygon_angle_deg'] is not None else None}}
            stages['polygon']['selection_deferred_to_local_lines']=self.proposal_policy=='multiscale' and mode!='rgb'
            stages['local_lines']=({'status':'computed','xy':fitted[0].tolist(),
                'angle_deg':fitted[1],'residual_px':fitted[2]} if fitted is not None else
                {'status':'insufficient_input','reason':'too_little_nondegenerate_line_support'})
            local_convex=None
            if fitted is not None:
                a,b=np.asarray(fitted[3])-fitted[0]
                orientation=np.sign(cv2.contourArea(contours[item['contour_id']],oriented=True))
                local_convex=bool((-a[0]*b[1]+a[1]*b[0])*orientation>0)
            stages['local_lines'].update(local_convex=local_convex, support=support, min_fit_support=self.min_fit_support,
                both_fits_available=fitted is not None and short is not None, checks=checks,
                used_for_selection=mode!='rgb')
            item['warnings']=['angle_stability_exceeds_diagnostic_threshold'] if (mode!='rgb'
                and self.angle_stability_mode=='diagnostic' and checks['angle_stability_deg']['passed'] is False) else []
            if mode=='depth':
                probe={'xy':item['xy'],'evidence':{},'status':'accepted','reasons':[]}
                self._depth_gate(probe,z,valid,component_labels,evidence['component_id'],interior,outer,intrinsics)
                stages['depth']={'status':'computed' if 'image_z_residual_p95_m' in probe['evidence'] else 'insufficient_input',
                    'candidate_quality':probe['status'],'warnings':probe['reasons'],'measurements':probe['evidence']}
            else:stages['depth']={'status':'insufficient_input','reason':'depth_stage_not_requested'}
            item['stage_results']=stages
        if self.proposal_policy=='multiscale':
            # Distinct polygon vertices can converge to the same fitted corner.
            # Keep one quality-ranked result within each component-local radius.
            unique=[]
            priority={'accepted':0,'unknown':1,'suspected':2}
            for item in sorted(candidates,key=lambda c:(priority[c['status']],
                    c['evidence'].get('line_residual_px',float('inf')),c['xy'])):
                match=next((q for q in unique if q['contour_id']==item['contour_id'] and
                    np.linalg.norm(np.asarray(q['xy'])-item['xy'])<=self.proposal_merge_px),None)
                if match is None:unique.append(item)
                else:
                    match['evidence']['proposal_scales']=sorted(set(
                        match['evidence']['proposal_scales']+item['evidence']['proposal_scales']))
            candidates=unique
        summary={'mode':mode,'support_policy':self.support_policy,'mask_source':provenance,'registration':'caller_declared',
                 'validation_status':'not_validated','physical_angle_verified':False,
                 'single_layer_guarantee':False,'counts':{s:sum(c['status']==s for c in candidates) for s in ('accepted','suspected','unknown')},
                 'depth_geometry':'camera_3d_plane' if intrinsics else 'local_image_z_trend',
                 'tuning':dict(min_area_px=float(self.min_area_px), polygon_epsilon_px=float(self.polygon_epsilon_px),
                     edge_window_px=self.edge_window_px, depth_radius_px=self.depth_radius_px,
                     proposal_policy=self.proposal_policy, support_policy=self.support_policy,
                     proposal_merge_px=self.proposal_merge_px, color_distance=float(self.color_distance),
                     proposal_angle_tolerance_deg=self.proposal_angle_tolerance_deg,
                     fitted_angle_tolerance_deg=self.fitted_angle_tolerance_deg,
                     angle_stability_mode=self.angle_stability_mode, angle_stability_deg=self.angle_stability_deg,
                     max_intersection_shift_px=self.max_intersection_shift_px, min_fit_support=self.min_fit_support,
                     line_residual_px=float(self.line_residual_px), depth_step_m=float(self.depth_step_m),
                     boundary_mixing_m=self.boundary_mixing_m, depth_residual_m=float(self.depth_residual_m)),
                 'depth_scale_m':float(depth_scale_m),'contours_considered':len({c['contour_id'] for c in candidates})}
        if self.proposal_policy=='multiscale':
            summary.update(proposal_policy='multiscale',proposal_epsilon_scales=[.5,1.,2.,4.],proposal_merge_px=self.proposal_merge_px)
        return CornerResult(candidates,summary,mask,{'contours_xy':debug_contours,'interior_mask':interior,'valid_depth_mask':valid,'component_labels':component_labels} if diagnostics else {})

    def _depth_gate(self,item,z,valid,component_labels,component_id,interior,outer,intrinsics):
        x,y=item['xy'];radius=self.depth_radius_px;h,w=z.shape
        x0,x1=max(0,int(np.floor(x-radius))),min(w,int(np.ceil(x+radius))+1)
        y0,y1=max(0,int(np.floor(y-radius))),min(h,int(np.ceil(y+radius))+1)
        yy,xx=np.mgrid[y0:y1,x0:x1];disk=(xx-x)**2+(yy-y)**2<=radius**2
        zi=z[y0:y1,x0:x1];ok=valid[y0:y1,x0:x1]
        labels=component_labels[y0:y1,x0:x1]
        own=labels==component_id
        # Preserve original holes; another component is not an internal fold.
        inside=interior[y0:y1,x0:x1]&own&disk
        band=own&~interior[y0:y1,x0:x1]&disk
        nearby=(labels>0)&~own&disk
        item['evidence'].update(nearby_other_component_pixels=int(nearby.sum()),
                                nearby_other_component_valid_depth_pixels=int((nearby&ok).sum()))
        count=int(inside.sum()); used=inside&ok
        fraction=float(used.sum()/count) if count else 0.
        edge_fraction=float((band&ok).sum()/band.sum()) if band.any() else 0.
        item['evidence'].update(interior_points=int(used.sum()),depth_valid_fraction=fraction,boundary_valid_fraction=edge_fraction)
        item['evidence']['depth_checks']={
            'interior_points':dict(value=int(used.sum()),minimum=20,passed=bool(used.sum()>=20)),
            'interior_valid_fraction':dict(value=fraction,minimum=.8,passed=bool(fraction>=.8)),
            'boundary_valid_fraction':dict(value=edge_fraction,minimum=.8,passed=bool(edge_fraction>=.8))}
        insufficient=used.sum()<20 or fraction<.8 or edge_fraction<.8
        if used.sum()<3:
            item.update(status='unknown',reasons=['insufficient_original_depth']);return
        design=np.column_stack((xx[used]-x,yy[used]-y,np.ones(int(used.sum()))))
        if np.linalg.matrix_rank(design)<3:
            item.update(status='unknown',reasons=['degenerate_depth_patch']);return
        coefficients=np.linalg.lstsq(design,zi[used],rcond=None)[0]
        predicted=coefficients[0]*(xx-x)+coefficients[1]*(yy-y)+coefficients[2]
        residual=np.abs(zi[used]-predicted[used])
        item['evidence']['image_z_residual_p95_m']=float(np.quantile(residual,.95))
        if intrinsics is not None:
            pts=np.column_stack(((xx[used]-intrinsics.cx)*zi[used]/intrinsics.fx,
                                 (yy[used]-intrinsics.cy)*zi[used]/intrinsics.fy,zi[used]))
            centered=pts-pts.mean(axis=0);_,singular,axes=np.linalg.svd(centered,full_matrices=False)
            if singular[1]<1e-8:
                item.update(status='unknown',reasons=['degenerate_camera_plane']);return
            residual=np.abs(centered@axes[-1])
        p95=float(np.quantile(residual,.95)); jumps=[]
        for pair,first,second,slope in (
                (used[:,1:]&used[:,:-1],zi[:,1:],zi[:,:-1],coefficients[0]),
                (used[1:,:]&used[:-1,:],zi[1:,:],zi[:-1,:],coefficients[1])):
            # Subtract only original valid pairs: missing infinities are not steps.
            jumps.extend(np.abs(first[pair]-second[pair]-slope).tolist())
        step_count=int(np.count_nonzero(np.array(jumps)>self.depth_step_m))
        outside=outer[y0:y1,x0:x1]&disk&ok
        external=float(np.median(zi[outside])-np.median(zi[used])) if outside.any() else None
        boundary_error=float(np.quantile(np.abs(zi[band&ok]-predicted[band&ok]),.95)) if (band&ok).any() else None
        item['evidence'].update(local_residual_p95_m=p95,internal_step_pair_count=step_count,
                                outside_minus_inside_z_m=external,boundary_trend_error_p95_m=boundary_error)
        item['evidence']['depth_checks'].update({
            'internal_step_pairs':dict(value=step_count,minimum_rejected=3,step_threshold_m=float(self.depth_step_m),passed=step_count<3),
            'local_residual_p95_m':dict(value=p95,maximum=float(self.depth_residual_m),passed=p95<=self.depth_residual_m),
            'boundary_trend_error_p95_m':dict(value=boundary_error,maximum=self.boundary_mixing_m,
                passed=boundary_error<=self.boundary_mixing_m if boundary_error is not None else None)})
        if insufficient:
            item.update(status='unknown',reasons=['insufficient_original_depth']);return
        if step_count>=3 or p95>self.depth_residual_m:
            item.update(status='suspected',reasons=['internal_depth_step_or_nonplanarity']);return
        if boundary_error is None or boundary_error>self.boundary_mixing_m:
            item.update(status='unknown',reasons=['boundary_depth_mixing_or_registration']);return
        item['reasons']=['visible_corner_no_local_depth_anomaly_hidden_layers_unresolved']
