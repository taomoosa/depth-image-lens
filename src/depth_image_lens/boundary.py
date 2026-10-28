"""Explainable color and depth boundary evidence for a selected overhead region."""
from dataclasses import dataclass, field
import numpy as np
from .debug import DebugResultMixin, _opencv
from .geometry import CameraIntrinsics, bool_mask
from .depth import _numeric_image, _validate_depth_scale


def _erode(mask, radius):
    cv=_opencv()
    return cv.erode(mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),
                    borderType=cv.BORDER_CONSTANT,borderValue=0)>0 if radius else mask.copy()


def _components(mask):
    cv=_opencv()
    count,labels,stats,_=cv.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return labels,[(i,int(stats[i,cv.CC_STAT_AREA])) for i in range(1,count)]


def _contours(mask):
    cv=_opencv();cs,_=cv.findContours(mask.astype(np.uint8),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    return [c[:,0,:].tolist() for c in cs if len(c)>=2]


def _inverse_plane(z, valid):
    """Fit 1/Z = ax+by+c; deterministic robust trimming, not a table calibration."""
    y,x=np.nonzero(valid)
    if len(x)<12:raise ValueError('insufficient_plane_support')
    stride=max(1,int(np.ceil(len(x)/5000)));x=x[::stride];y=y[::stride]
    design=np.c_[x,y,np.ones(len(x))];target=1./z[y,x];keep=np.ones(len(x),bool)
    for _ in range(4):
        coeff,_,rank,_=np.linalg.lstsq(design[keep],target[keep],rcond=None)
        if rank<3:raise ValueError('degenerate_plane_support')
        residual=target-design@coeff;center=np.median(residual[keep])
        mad=np.median(np.abs(residual[keep]-center));next_keep=np.abs(residual-center)<=max(3*1.4826*mad,1e-9)
        if next_keep.sum()<12:break
        keep=next_keep
    coeff,_,rank,_=np.linalg.lstsq(design[keep],target[keep],rcond=None)
    if rank<3 or not np.isfinite(coeff).all():raise ValueError('degenerate_plane_support')
    return coeff,int(keep.sum())


def _background_selection(rgb, lab, background, color_order, valid_mask, check_mask,
                          shift_xy, parameters, object_color_seed=None):
    """Registered empty-reference evidence; check ROI cannot certify unseen pixels."""
    cv = _opencv()
    shape = rgb.shape[:2]
    bg = np.asarray(background)
    if bg.dtype != np.uint8 or bg.shape != rgb.shape:
        raise ValueError('background_color must match current uint8 HxWx3')
    shift = np.asarray(shift_xy)
    if shift.shape != (2,) or shift.dtype.kind not in 'iu':
        raise ValueError('background_shift_xy must contain two integer pixels')
    dx, dy = map(int, shift)
    if abs(dx) >= shape[1] or abs(dy) >= shape[0]:
        raise ValueError('background translation removes image overlap')
    available = np.ones(shape, bool) if valid_mask is None else bool_mask(valid_mask, shape, 'background_valid_mask').copy()
    bg = bg if color_order == 'RGB' else bg[..., ::-1]
    if dx or dy:
        matrix = np.array([[1., 0., dx], [0., 1., dy]])
        bg = cv.warpAffine(bg, matrix, (shape[1], shape[0]), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT)
        available = cv.warpAffine(available.astype(np.uint8), matrix, (shape[1], shape[0]), flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT) > 0
    ref_lab = cv.cvtColor(bg, cv.COLOR_RGB2LAB).astype(float)
    check = np.zeros(shape, bool) if check_mask is None else bool_mask(check_mask, shape, 'background_check_mask').copy()
    samples = check & available
    offset = np.zeros(3)
    raw_p95 = corrected_p95 = None
    rejection = None
    if available.any() and (check_mask is not None or parameters['background_compensate']):
        if samples.sum() < 25:
            raise ValueError('background check needs at least 25 valid declared table pixels')
        delta = lab[samples] - ref_lab[samples]
        raw_p95 = float(np.quantile(np.linalg.norm(delta, axis=1), .95))
        if parameters['background_compensate']:
            offset = np.median(delta, axis=0)
            if np.linalg.norm(offset) > parameters['background_max_offset']:
                rejection = 'illumination_offset_exceeds_limit'
        corrected_p95 = float(np.quantile(np.linalg.norm(delta-offset, axis=1), .95))
        if corrected_p95 > parameters['background_check_p95']:
            rejection = rejection or 'check_residual_exceeds_limit'
    distance = np.linalg.norm(lab-(ref_lab+offset), axis=2)
    distance[~available] = np.nan
    obj = available & (distance > parameters['color_distance'])
    uncertain = ~available | (np.abs(distance-parameters['color_distance']) <= parameters['color_margin'])
    info = dict(status='checked_candidate' if samples.any() else 'unchecked_candidate',
                reference_valid_fraction=float(available.mean()), shift_xy=[dx, dy],
                check_pixels=int(samples.sum()), check_raw_p95=raw_p95,
                check_corrected_p95=corrected_p95, encoded_lab_offset=offset.tolist(),
                compensation_applied=bool(parameters['background_compensate'] and samples.any()),
                check_threshold=parameters['background_check_p95'],
                warnings=['empty_reference_and_alignment_are_caller_assumptions',
                          'check_region_does_not_validate_covered_or_contaminated_reference_regions',
                          'shadows_and_same_color_objects_can_defeat_color_difference'])
    if rejection:info.update(status='rejected',reason=rejection,fallback='border_table_assumption')
    if not available.any():info.update(status='insufficient', reason='no_valid_background_pixels')
    debug=dict(background_valid_mask=available,background_check_mask=check,background_distance=distance)
    if parameters['background_difference_mode']=='separate_tolerances':
        from ._background_difference import separate_lab_selection
        clipped=(rgb.min(axis=2)<=1)|(rgb.max(axis=2)>=254)|(bg.min(axis=2)<=1)|(bg.max(axis=2)>=254)
        obj,uncertain,difference,extra=separate_lab_selection(lab-(ref_lab+offset),available,clipped,parameters)
        difference['applied_to_selection']=not rejection and bool(available.any())
        info['difference_model']=difference;debug.update(extra)
    if parameters['background_shadow_mode']=='uniform_color':
        if rejection or not available.any():
            info['shadow_separation']=dict(status='insufficient',reasons=['background_reference_not_usable_for_shadow_suppression'])
        else:
            from ._background_shadow import uniform_shadow_selection
            corrected=bg if not offset.any() else cv.cvtColor(np.clip(np.rint(ref_lab+offset),0,255).astype(np.uint8),cv.COLOR_LAB2RGB)
            obj,uncertain,shadow,extra=uniform_shadow_selection(rgb,corrected,available,check,object_color_seed,obj,uncertain,parameters)
            info['shadow_separation']=shadow;debug.update(extra)
    return obj, uncertain, distance, info, debug


@dataclass(frozen=True)
class BoundaryResult(DebugResultMixin):
    stages: dict
    metrics: dict
    parameters: dict
    object_mask: np.ndarray = field(repr=False)
    uncertain_mask: np.ndarray = field(repr=False)
    color_boundary_mask: np.ndarray = field(repr=False)
    depth_boundary_mask: np.ndarray = field(repr=False)
    agreement_mask: np.ndarray = field(repr=False)
    island_mask: np.ndarray = field(repr=False)
    depth_domain_mask: np.ndarray = field(repr=False)
    darkness: np.ndarray = field(repr=False)
    gradient: np.ndarray = field(repr=False)
    depth_residual_m: np.ndarray = field(repr=False)
    exterior: list
    lines: list
    islands: list
    diagnostics: dict = field(default_factory=dict,repr=False)
    layer_status: str = 'unknown'
    scope: str = 'color_and_relative_depth_evidence_not_semantic_fold_or_grasp_truth'
    topology: dict = field(default_factory=dict)

    def to_dict(self):
        return {k:v for k,v in vars(self).items() if not isinstance(v,np.ndarray) and k!='diagnostics'}

    def filter_outer_proximity(self,*,near_px=8.,reject_fraction=.7,include_holes=False,
                               observed_mask=None):
        """Classify topology candidates without changing the original graph or masks."""
        from .boundary_proximity import filter_outer_proximity
        return filter_outer_proximity(self,near_px=near_px,reject_fraction=reject_fraction,
                                      include_holes=include_holes,observed_mask=observed_mask)

    def render_debug(self,color=None,*,color_order='RGB',depth=None,depth_scale_m=.001,topology_labels=False,**options):
        """Existing mask panels plus fixed-scale darkness, gradient and residual heatmaps."""
        if type(topology_labels) is not bool:raise ValueError('topology_labels must be bool')
        base=super().render_debug(color,color_order=color_order,depth=depth,depth_scale_m=depth_scale_m,**options)
        cv=_opencv();tiles=[];width=max(100,base.shape[1]//3)
        limit=max(self.parameters['island_height_m']*4,1e-9)
        values=[('Darkness 0..255',self.darkness,255.,False),
                ('Gradient 0..255 per pixel',self.gradient,255.,False),
                (f'Relative depth +/-{limit:g}m',self.depth_residual_m,limit,True)]
        if 'background_distance' in self.diagnostics:
            values.append(('Background Lab distance 0..90',self.diagnostics['background_distance'],90.,False))
        for key,title,scale in [('background_chroma_distance','Background chroma distance 0..90',90.),('background_luma_distance','Background luma distance 0..90',90.),('background_combined_score','Background score 0..4; threshold 1',4.)]:
            if key in self.diagnostics:values.append((title,self.diagnostics[key],scale,False))
        for key,title,scale in [('shadow_chroma_distance','RGB chromaticity distance 0..0.2',.2),('shadow_intensity_ratio','Current/reference intensity ratio 0..2',2.),('shadow_suppressed_mask','Suppressed shadow hypothesis 0/1',1.),('shadow_unreliable_mask','Unreliable photometric support 0/1',1.),('object_color_seed_mask','Declared object color sample 0/1',1.)]:
            if key in self.diagnostics:values.append((title,self.diagnostics[key],scale,False))
        for title,a,scale,signed in values:
            good=np.isfinite(a);u=np.zeros(a.shape,np.uint8)
            u[good]=np.clip((a[good]/scale*(.5 if signed else 1)+(.5 if signed else 0))*255,0,255).astype(np.uint8)
            im=cv.applyColorMap(u,cv.COLORMAP_TURBO)
            if color_order=='RGB':im=im[...,::-1].copy()
            im[~good]=(255,0,255)
            h=max(1,round(im.shape[0]*width/im.shape[1]));im=cv.resize(im,(width,h),interpolation=cv.INTER_NEAREST)
            tile=np.zeros((h+28,width,3),np.uint8);tile[28:]=im;cv.putText(tile,title,(4,18),cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);tiles.append(tile)
        tile_height=max(t.shape[0] for t in tiles)
        strip=np.zeros((tile_height*((len(tiles)+2)//3),base.shape[1],3),np.uint8)
        for i,t in enumerate(tiles):
            x=(i%3)*width;y=(i//3)*tile_height;end=min(x+width,strip.shape[1]);strip[y:y+t.shape[0],x:end]=t[:,:end-x]
        panels=[base,strip]
        if self.topology:
            from .debug import depth_canvas
            source=np.array(color,copy=True) if color is not None else depth_canvas(depth,depth_scale_m) if depth is not None else np.full((*self.object_mask.shape,3),35,np.uint8)
            if color is None and color_order=='BGR':source=source[...,::-1].copy()
            height=max(1,round(source.shape[0]*width/source.shape[1]));row=np.zeros((max(height+50,310),base.shape[1],3),np.uint8)
            for i,kind in enumerate(('color','depth')):
                graph=self.topology[kind];tile=source.copy()
                inks={'strong':(20,230,80),'weak':(60,170,255),'inferred':(255,150,20)}
                for segment in graph['segments']:
                    for (x,y),evidence in zip(segment['points_xy'],segment['evidence']):tile[y,x]=inks[evidence] if color_order=='RGB' else inks[evidence][::-1]
                for node in graph['nodes']:
                    ink=(255,40,180) if node['kind']=='corner' else (255,255,40) if node['kind']=='junction' else (220,220,220)
                    ink=ink if color_order=='RGB' else ink[::-1]
                    cv.circle(tile,tuple(node['xy']),2,ink,1)
                    if topology_labels:cv.putText(tile,str(node['id']),tuple(node['xy']),cv.FONT_HERSHEY_SIMPLEX,.3,ink,1)
                for group in graph.get('junction_grouping',{}).get('groups',[]):
                    center=tuple(np.rint(group['center_xy']).astype(int));ink=(40,255,255) if color_order=='RGB' else (255,255,40)
                    cv.circle(tile,center,5,ink,1)
                x=i*width;end=min(x+width,row.shape[1]);row[50:50+height,x:end]=cv.resize(tile,(width,height),interpolation=cv.INTER_NEAREST)[:,:end-x]
                cv.putText(row,kind+' topology: '+graph['status'],(x+3,16),cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                cv.putText(row,'green strong / blue weak / orange inferred',(x+3,33),cv.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
            settings=['color_weak_ratio','depth_weak_ratio','max_gap_px','gap_angle_deg','tangent_window_px','corner_turn_deg','corner_window_px','min_path_length_px','junction_cluster_px','junction_cluster_residual_px','junction_cluster_angle_deg']
            for j,key in enumerate(settings):
                cv.putText(row,f'{key}: {self.parameters[key]}',(2*width+4,20+20*j),cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv.putText(row,'magenta corner / yellow junction',(2*width+4,260),cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            cv.putText(row,'cyan ring: inferred junction group',(2*width+4,280),cv.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            panels.append(row)
        return np.concatenate(panels,axis=0)


class OverheadBoundaryDetector:
    """Configurable foreground and boundary evidence; no learned semantic model.

    Color hypotheses can survive unavailable depth. Depth evidence can use an
    explicit depth_roi when color selection is ambiguous. Layer status is unknown.
    """
    def __init__(self,*,color_distance=30.,color_margin=8.,min_object_area=100,
                 border_px=5,boundary_guard_px=3,color_window_px=9,
                 darkness_threshold=12.,gradient_threshold=12.,min_line_pixels=6,
                 depth_step_m=.01,depth_noise_m=.002,island_height_m=.01,
                 min_island_area=25,max_island_fraction=.35,association_px=3,island_context_px=5,grabcut_iterations=3,
                 background_compensate=False,background_check_p95=20.,background_max_offset=40.,
                 background_difference_mode='lab_distance',background_chroma_tolerance=30.,
                 background_luma_tolerance=30.,background_score_margin=8./30.,
                 background_shadow_mode='off',shadow_chroma_threshold=.04,shadow_intensity_floor=20.,
                 shadow_min_ratio=.35,shadow_max_ratio=1.15,
                 trace_boundaries=False,color_weak_ratio=1.,depth_weak_ratio=1.,max_gap_px=0,
                 gap_angle_deg=25.,corner_turn_deg=45.,corner_window_px=5,tangent_window_px=5,min_path_length_px=3.,
                 junction_cluster_px=0,junction_cluster_residual_px=1.,junction_cluster_angle_deg=15.):
        ints=dict(min_object_area=min_object_area,border_px=border_px,boundary_guard_px=boundary_guard_px,
                  color_window_px=color_window_px,min_line_pixels=min_line_pixels,min_island_area=min_island_area,association_px=association_px,island_context_px=island_context_px,grabcut_iterations=grabcut_iterations)
        if any(type(v) is not int or v<0 for v in ints.values()) or min(min_object_area,border_px,min_line_pixels,min_island_area,island_context_px)<1:
            raise ValueError('nonnegative integer pixel settings; areas, border and line support positive')
        if not 1<=grabcut_iterations<=10:raise ValueError('grabcut_iterations must be 1..10')
        if color_window_px<3 or color_window_px%2!=1 or color_window_px>101:raise ValueError('color_window_px must be odd,3..101')
        if type(background_compensate) is not bool:raise ValueError('background_compensate must be bool')
        if not np.isfinite([background_check_p95,background_max_offset]).all() or min(background_check_p95,background_max_offset)<=0:raise ValueError('positive finite background thresholds required')
        if background_difference_mode not in ('lab_distance','separate_tolerances'):
            raise ValueError('background_difference_mode must be lab_distance or separate_tolerances')
        difference_values = [background_chroma_tolerance, background_luma_tolerance, background_score_margin]
        try:
            if any(isinstance(v, (bool, np.bool_)) for v in difference_values):raise ValueError
            tc, tl, score_margin = map(float, difference_values)
            if not np.isfinite([tc, tl, score_margin]).all() or tc<=0 or tl<=0 or not 0<=score_margin<=1:raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ValueError('finite positive background tolerances and radius margin in [0,1] required') from None
        difference_parameters = dict(background_difference_mode=background_difference_mode,
                                     background_chroma_tolerance=tc,background_luma_tolerance=tl,
                                     background_score_margin=score_margin)
        if background_shadow_mode not in ('off','uniform_color'):raise ValueError('background_shadow_mode must be off or uniform_color')
        if not np.isfinite([shadow_chroma_threshold,shadow_intensity_floor,shadow_min_ratio,shadow_max_ratio]).all() or not(0<shadow_chroma_threshold<=.2 and 1<=shadow_intensity_floor<254 and 0<shadow_min_ratio<=1<=shadow_max_ratio<=2):raise ValueError('invalid shadow chromaticity, intensity or ratio setting')
        shadow_parameters=dict(background_shadow_mode=background_shadow_mode,shadow_chroma_threshold=float(shadow_chroma_threshold),shadow_intensity_floor=float(shadow_intensity_floor),shadow_min_ratio=float(shadow_min_ratio),shadow_max_ratio=float(shadow_max_ratio))
        if type(trace_boundaries) is not bool:raise ValueError('trace_boundaries must be bool')
        if type(max_gap_px) is not int or not 0<=max_gap_px<=20:raise ValueError('max_gap_px must be integer 0..20')
        if type(tangent_window_px) is not int or not 2<=tangent_window_px<=50:raise ValueError('tangent_window_px must be integer 2..50')
        if type(corner_window_px) is not int or not 2<=corner_window_px<=50:raise ValueError('corner_window_px must be integer 2..50')
        if not np.isfinite([color_weak_ratio,depth_weak_ratio,gap_angle_deg,corner_turn_deg,min_path_length_px]).all() or not(0<color_weak_ratio<=1 and 0<depth_weak_ratio<=1 and 0<=gap_angle_deg<=60 and 0<corner_turn_deg<=180 and min_path_length_px>=0):raise ValueError('invalid topology ratio, angle or length setting')
        if type(junction_cluster_px) is not int or not 0<=junction_cluster_px<=20:raise ValueError('junction_cluster_px must be integer 0..20')
        if not np.isfinite([junction_cluster_residual_px,junction_cluster_angle_deg]).all() or not(0<junction_cluster_residual_px<=3 and 0<=junction_cluster_angle_deg<=45):raise ValueError('junction residual must be in(0,3]px and angle in[0,45]deg')
        topology_parameters=dict(junction_cluster_px=junction_cluster_px,junction_cluster_residual_px=float(junction_cluster_residual_px),junction_cluster_angle_deg=float(junction_cluster_angle_deg),trace_boundaries=trace_boundaries,color_weak_ratio=float(color_weak_ratio),depth_weak_ratio=float(depth_weak_ratio),max_gap_px=max_gap_px,gap_angle_deg=float(gap_angle_deg),corner_turn_deg=float(corner_turn_deg),corner_window_px=corner_window_px,tangent_window_px=tangent_window_px,min_path_length_px=float(min_path_length_px))
        values=[color_distance,color_margin,darkness_threshold,gradient_threshold,depth_step_m,depth_noise_m,island_height_m,max_island_fraction]
        if not np.isfinite(values).all() or min(values[:-1])<=0 or not 0<max_island_fraction<1:raise ValueError('finite positive thresholds and island fraction in(0,1) required')
        self.parameters=dict(ints,color_distance=float(color_distance),color_margin=float(color_margin),darkness_threshold=float(darkness_threshold),
            gradient_threshold=float(gradient_threshold),depth_step_m=float(depth_step_m),depth_noise_m=float(depth_noise_m),island_height_m=float(island_height_m),max_island_fraction=float(max_island_fraction),background_compensate=background_compensate,background_check_p95=float(background_check_p95),background_max_offset=float(background_max_offset),**topology_parameters,**shadow_parameters,**difference_parameters)

    @classmethod
    def from_config(cls,config):
        """Construct from a JSON-compatible parameter dictionary; unknown keys rejected."""
        return cls(**dict(config))

    def detect(self,color,depth=None,*,color_order,registered_to_depth=False,depth_scale_m=.001,
               object_mask=None,background_color=None,table_mask=None,object_roi=None,depth_roi=None,
               intrinsics=None,observed_mask=None,diagnostics=False,background_valid_mask=None,
               background_check_mask=None,background_shift_xy=(0,0),object_color_seed=None):
        """Color uint8 HxWx3; optional rectified optical-axis Z and bool HxW masks.

        Choose one segmentation source; explicit object_mask overrides optional background. With none,
        border pixels are assumed table (reported, not verified). Masks override
        only segmentation, never supply wrinkle truth. Reference color requires
        registered empty-scene provenance; optional shift and declared-table offset
        correction do not certify emptiness. Invalid optional references fall back
        with uncertainty. Scale is metres/count for every dtype.
        """
        cv=_opencv();rgb=np.asarray(color)
        if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8 or min(rgb.shape[:2])<3:raise ValueError('color must be uint8 HxWx3')
        if color_order not in ('RGB','BGR'):raise ValueError('explicit RGB or BGR required')
        if type(diagnostics) is not bool:raise ValueError('diagnostics must be bool')
        if sum(v is not None for v in (object_mask,table_mask,object_roi))>1 or (background_color is not None and (table_mask is not None or object_roi is not None)):
            raise ValueError('choose one segmentation source; only object_mask may override background')
        shape=rgb.shape[:2];p=self.parameters
        rgb=rgb if color_order=='RGB' else rgb[...,::-1];lab=cv.cvtColor(rgb,cv.COLOR_RGB2LAB).astype(np.float32)
        uncertain=np.zeros(shape,bool);reasons=[];background_info=None;reference_debug={};reference_result=None
        if background_color is not None:
            if object_mask is not None:
                background_info=dict(status='ignored',reason='explicit_object_mask_has_priority')
            else:
                try:
                    reference_result=_background_selection(rgb,lab,background_color,color_order,background_valid_mask,background_check_mask,background_shift_xy,p,object_color_seed=object_color_seed)
                    if reference_result[3]['status']=='rejected':
                        background_info=reference_result[3];reference_debug=reference_result[4]
                        background_color=None;reasons.append('optional_background_rejected_using_uncertain_border_fallback')
                except (ValueError,TypeError,cv.error) as exc:
                    background_info=dict(status='insufficient',reason='invalid_or_rejected_background',error=str(exc),fallback='border_table_assumption')
                    reasons.append('optional_background_rejected_using_uncertain_border_fallback')
                    background_color=None
        if object_mask is not None:
            obj=bool_mask(object_mask,shape,'object_mask').copy();mode='supplied_mask';sep=None
        elif object_roi is not None:
            roi=np.asarray(object_roi)
            if roi.shape!=(4,) or roi.dtype.kind not in 'iu':raise ValueError('object_roi requires integer xyxy')
            x0,y0,x1,y1=map(int,roi);h,w=shape
            if not (0<=x0<x1<=w and 0<=y0<y1<=h) or (x1-x0)*(y1-y0)>=h*w:raise ValueError('object_roi must leave observed background outside rectangle')
            seed=np.full(shape,cv.GC_BGD,np.uint8);seed[y0:y1,x0:x1]=cv.GC_PR_FGD
            mode='rectangle_grabcut';sep=None;obj=np.zeros(shape,bool)
            try:
                cv.grabCut(np.ascontiguousarray(rgb),seed,None,np.zeros((1,65),float),np.zeros((1,65),float),p['grabcut_iterations'],cv.GC_INIT_WITH_MASK)
                obj=(seed==cv.GC_FGD)|(seed==cv.GC_PR_FGD)
                uncertain[y0:y1,x0:x1]=True
                reasons.extend(['rectangle_and_outside_background_are_caller_assumptions','grabcut_probable_labels_are_not_semantic_confidence','opencv_initialization_may_vary_use_recorded_process_rng_for_replay'])
            except cv.error:
                uncertain[y0:y1,x0:x1]=True;reasons.append('grabcut_insufficient_color_model')
        elif background_color is not None:
            obj,uncertain,dist,background_info,reference_debug=reference_result
            mode='registered_background_color';sep=float(np.median(dist[obj])) if obj.any() else None
            reasons.append('background_alignment_and_lighting_are_caller_assumptions')
        else:
            if table_mask is None:
                seed=np.zeros(shape,bool);b=min(p['border_px'],min(shape)//2);seed[:b]=True;seed[-b:]=True;seed[:,:b]=True;seed[:,-b:]=True;mode='border_table_assumption'
            else:seed=bool_mask(table_mask,shape,'table_mask');mode='table_color_seed'
            if seed.sum()<3:
                obj=np.zeros(shape,bool);uncertain[:]=True;sep=None;reasons.append('insufficient_table_color_samples')
            else:
                # Most frequent quantized table colors, without a skin/object prior.
                bins,counts=np.unique((lab[seed]/8).astype(np.int16),axis=0,return_counts=True)
                palette=bins[np.argsort(-counts,kind='stable')[:16]]*8+4
                dist=np.full(shape,np.inf)
                for sample in palette:dist=np.minimum(dist,np.linalg.norm(lab-sample,axis=2))
                obj=dist>p['color_distance'];uncertain=np.abs(dist-p['color_distance'])<=p['color_margin'];sep=float(np.median(dist[obj])) if obj.any() else None
                reasons.append('table_appearance_can_match_object_or_change_with_shadows')
        if background_info is not None and background_info.get('fallback'):
            uncertain[:]=True
        labels,parts=_components(obj)
        for label,n in parts:
            if n<p['min_object_area']:obj[labels==label]=False;uncertain[labels==label]=True
        if not obj.any():reasons.append('no_color_separation_is_not_proof_of_no_object')
        if np.any(obj[0]) or np.any(obj[-1]) or np.any(obj[:,0]) or np.any(obj[:,-1]):reasons.append('selection_touches_image_boundary')
        stages={'segmentation':dict(status='candidate' if obj.any() else 'insufficient',mode=mode,reasons=reasons)}
        if background_info is not None:stages['background_reference']=background_info
        if p['background_difference_mode']=='separate_tolerances':
            stages['background_difference']=(background_info or {}).get('difference_model',dict(
                applied_to_selection=False,reasons=['explicit_object_mask_has_priority' if object_mask is not None else 'usable_background_reference_required']))
        if p['background_shadow_mode']=='uniform_color':
            stages['shadow_separation']=(background_info or {}).get('shadow_separation',dict(
                status='ignored' if object_mask is not None else 'insufficient',
                reasons=['explicit_object_mask_has_priority' if object_mask is not None else 'usable_background_reference_required']))
        gray=cv.cvtColor(rgb,cv.COLOR_RGB2GRAY).astype(np.float32)
        close=cv.morphologyEx(gray,cv.MORPH_CLOSE,np.ones((p['color_window_px'],p['color_window_px']),np.uint8))
        darkness=np.maximum(0,close-gray);gx=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)/8;gy=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3)/8
        gradient=np.hypot(gx,gy);inside=_erode(obj,max(p['boundary_guard_px'],p['color_window_px']//2))
        colors=inside&((darkness>=p['darkness_threshold'])|(gradient>=p['gradient_threshold']))
        labs,parts=_components(colors);lines=[]
        for label,n in parts:
            m=labs==label
            if n<p['min_line_pixels']:colors[m]=False;continue
            lines.append(dict(id=len(lines)+1,pixels=n,band_contours_xy=_contours(m),darkness_mean=float(darkness[m].mean()),darkness_p95=float(np.quantile(darkness[m],.95)),gradient_p95=float(np.quantile(gradient[m],.95)),interpretation='color_band_may_be_print_seam_shadow_or_geometric_boundary'))
        stages['color']=dict(status='computed' if obj.any() else 'insufficient',reason='literal_contrast_not_semantic_fold',line_count=len(lines))
        domain=bool_mask(depth_roi,shape,'depth_roi').copy() if depth_roi is not None else obj.copy()
        domain_source='caller_depth_roi' if depth_roi is not None else 'color_or_supplied_object'
        # No object identity is fabricated when segmentation fails.
        if depth_roi is None and object_mask is None and not obj.any():domain[:]=True;domain_source='full_frame_without_object_identity'
        steps=np.zeros(shape,bool);island=np.zeros(shape,bool);residual=np.full(shape,np.nan);patches=[];valid=np.zeros(shape,bool);components=[]
        weak_steps=np.zeros(shape,bool);trace_valid=np.zeros(shape,bool);step_strength=np.full(shape,np.nan)
        depth_error=None
        if depth is not None:
            try:
                _validate_depth_scale(depth_scale_m)
                if registered_to_depth is not True:raise ValueError('registered_to_depth=True required when depth is supplied')
                if intrinsics is not None and not isinstance(intrinsics,CameraIntrinsics):raise ValueError('CameraIntrinsics required')
                raw=_numeric_image(depth,'depth')
                if raw.shape!=shape:raise ValueError('depth/color shape mismatch')
                with np.errstate(over='ignore',invalid='ignore'):z=raw*depth_scale_m
                valid=np.isfinite(z)&(z>0)
                if observed_mask is not None:valid &= bool_mask(observed_mask,shape,'observed_mask')
            except (ValueError,TypeError) as exc:
                depth_error=str(exc);valid=np.zeros(shape,bool)
        if depth is None or depth_error is not None:
            stages['depth']=dict(status='insufficient',reason='depth_not_supplied' if depth is None else 'invalid_depth_contract',error=depth_error,domain=domain_source)
        else:
            core=_erode(domain,p['boundary_guard_px']);labs,parts=_components(core)
            yy,xx=np.indices(shape)
            for label,n in parts:
                region=labs==label;ok=region&valid;item=dict(component=label,selected_pixels=n,valid_pixels=int(ok.sum()),status='insufficient')
                try:
                    coeff,support=_inverse_plane(z,ok);inv=coeff[0]*xx+coeff[1]*yy+coeff[2]
                    use=ok&np.isfinite(inv)&(inv>1e-9)
                    if use.sum()<12:raise ValueError('insufficient_positive_plane_predictions')
                    pred=np.zeros(shape);pred[use]=1./inv[use];r=np.full(shape,np.nan);r[use]=pred[use]-z[use]
                    if intrinsics is not None:
                        vector=np.array([coeff[0]*intrinsics.fx,coeff[1]*intrinsics.fy,coeff[0]*intrinsics.cx+coeff[1]*intrinsics.cy+coeff[2]])
                        norm=float(np.linalg.norm(vector))
                        if not np.isfinite(norm) or norm<1e-12:raise ValueError('invalid_plane_normal')
                        r[use]=(1-z[use]*inv[use])/norm
                    residual[use]=r[use]
                    # Detrended Z steps use only real adjacent samples of this component.
                    zr=np.full(shape,np.nan);zr[use]=pred[use]-z[use];local_steps=np.zeros(shape,bool)
                    threshold=max(p['depth_step_m'],2*p['depth_noise_m'])
                    for dy,dx in ((0,1),(1,0)):
                        a=np.s_[:shape[0]-dy or None,:shape[1]-dx or None];b=np.s_[dy:,dx:]
                        with np.errstate(invalid='ignore'):hit=use[a]&use[b]&(np.abs(zr[a]-zr[b])>=threshold)
                        local_steps[a]|=hit;local_steps[b]|=hit
                    steps |=local_steps
                    if p['trace_boundaries']:
                        trace_valid|=use
                        for dy,dx in ((0,1),(1,0)):
                            a=np.s_[:shape[0]-dy or None,:shape[1]-dx or None];b=np.s_[dy:,dx:]
                            with np.errstate(invalid='ignore'):delta=np.abs(zr[a]-zr[b])
                            pair_valid=use[a]&use[b]
                            hit=pair_valid&(delta>=max(p['depth_step_m']*p['depth_weak_ratio'],2*p['depth_noise_m']))
                            weak_steps[a]|=hit;weak_steps[b]|=hit
                            values=np.where(pair_valid,delta,np.nan)
                            step_strength[a]=np.fmax(step_strength[a],values);step_strength[b]=np.fmax(step_strength[b],values)
                    for sign in (1,-1):
                        selected=use&(sign*r>=max(p['island_height_m'],2*p['depth_noise_m']));il,ip=_components(selected)
                        for key,area in ip:
                            if area<p['min_island_area']:continue
                            m=il==key;ring=(cv.dilate(m.astype(np.uint8),np.ones((3,3),np.uint8))>0)&~m
                            contained=bool(ring.any() and np.all(use[ring]));fraction=area/max(1,int(use.sum()));accepted=contained and fraction<=p['max_island_fraction']
                            # Verify local contrast against observed surrounding samples,
                            # excluding the candidate itself from this reference fit.
                            radius=p['island_context_px']
                            annulus=(cv.dilate(m.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))>0)&~m&use
                            local_contrast=None;local_reason='insufficient_local_context'
                            try:
                                ac,context_support=_inverse_plane(z,annulus)
                                ai=ac[0]*xx[m]+ac[1]*yy[m]+ac[2]
                                if np.any(ai<=1e-9):raise ValueError('invalid_local_prediction')
                                ar=1./ai-z[m]
                                if intrinsics is not None:
                                    av=np.array([ac[0]*intrinsics.fx,ac[1]*intrinsics.fy,ac[0]*intrinsics.cx+ac[1]*intrinsics.cy+ac[2]])
                                    ar=(1-z[m]*ai)/np.linalg.norm(av)
                                local_contrast=float(np.median(ar))
                                local_reason='local_context_contrast' if sign*local_contrast>=max(p['island_height_m'],2*p['depth_noise_m']) else 'weak_local_context_contrast'
                            except (ValueError,np.linalg.LinAlgError,FloatingPointError):context_support=0
                            accepted=accepted and local_reason=='local_context_contrast'
                            boundary=_erode(m,1)^m;support_fraction=float((boundary&cv.dilate(local_steps.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)).sum()/max(1,boundary.sum()))
                            patches.append(dict(component=label,sign='toward_camera' if sign>0 else 'away_from_camera',area_px=area,component_fraction=fraction,median_residual_m=float(np.median(r[m])),local_context_contrast_m=local_contrast,local_context_support=context_support,local_context_reason=local_reason,boundary_step_fraction=support_fraction,contained_observed=contained,status='relative_depth_patch_candidate' if accepted else 'unknown',reason='local_relief_not_layer_count' if accepted else 'unobserved_boundary_or_broad_component_change',contours_xy=_contours(m)))
                            if accepted:island|=m
                    item.update(status='computed',fit_support=support,plane_inverse_z_coefficients=coeff.tolist(),residual_units='normal_m_toward_camera' if intrinsics is not None else 'camera_z_m_toward_camera',robust_residual_p95_m=float(np.quantile(np.abs(r[use]),.95)) if use.any() else None)
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:item['reason']=str(exc)
                components.append(item)
            stages['depth']=dict(status='computed' if any(c['status']=='computed' for c in components) else 'insufficient',domain=domain_source,components=components,reason='relative_component_plane_not_table_or_layer_truth')
        stages['depth']['unscoped_evidence']=domain_source=='full_frame_without_object_identity'
        stages['depth']['object_identity_available']=domain_source=='color_or_supplied_object' and bool(obj.any())
        near=cv.dilate(steps.astype(np.uint8),np.ones((2*p['association_px']+1,2*p['association_px']+1),np.uint8))>0
        agreement=colors&near
        metrics=dict(object_pixels=int(obj.sum()),uncertain_pixels=int(uncertain.sum()),color_separation_median=sep,color_boundary_pixels=int(colors.sum()),depth_boundary_pixels=int(steps.sum()),agreement_pixels=int(agreement.sum()),color_agreement_fraction=float(agreement.sum()/colors.sum()) if colors.any() else None,depth_valid_fraction=float((valid&domain).sum()/domain.sum()) if domain.any() else None,island_pixels=int(island.sum()),warnings=['color_is_not_semantic_boundary','single_sheet_relief_can_mimic_overlap','relative_plane_is_not_table_reference','islands_depend_on_component_baseline_and_observation','reference_color_and_seed_modes_do_not_remove_hands_semantically'])
        stages['association']=dict(status='computed' if colors.any() and steps.any() else 'insufficient',reason='spatial_proximity_not_causal_confirmation',tolerance_px=p['association_px'])
        stages['islands']=dict(status='computed' if any(c['status']=='computed' for c in components) else 'insufficient',candidate_count=sum(x['status']=='relative_depth_patch_candidate' for x in patches))
        topology={};trace_debug={}
        if p['trace_boundaries']:
            from ._boundary_topology import trace_evidence
            weak_color=inside&((darkness>=p['darkness_threshold']*p['color_weak_ratio'])|(gradient>=p['gradient_threshold']*p['color_weak_ratio']))
            for name,strong,weak,allowed,values in [('color',colors,weak_color,inside,dict(darkness=darkness,gradient=gradient)),('depth',steps,weak_steps,trace_valid,dict(step_strength_m=step_strength))]:
                graph,maps=trace_evidence(strong,weak,allowed,p,values)
                graph['domain']=stages['depth']['domain'] if name=='depth' else 'object_interior'
                if (name=='color' and not obj.any()) or (name=='depth' and stages['depth']['status']=='insufficient'):graph['status']='insufficient'
                topology[name]=graph
                if diagnostics:trace_debug[name+'_topology']=maps
        return BoundaryResult(stages,metrics,dict(p),obj,uncertain,colors,steps,agreement,island,domain,darkness,gradient,residual,_contours(obj),lines,patches,dict(valid_depth_mask=valid,**reference_debug,**trace_debug) if diagnostics else {},topology=topology)
