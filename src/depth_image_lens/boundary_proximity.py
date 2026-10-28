"""Non-destructive outer-proximity selection of boundary graph evidence."""
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from .debug import DebugResultMixin, _opencv, depth_canvas
from .geometry import bool_mask


def _json_value(value):
    if isinstance(value,dict):return {k:_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [_json_value(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


@dataclass(frozen=True)
class BoundaryProximityResult(DebugResultMixin):
    records: list
    parameters: dict
    summary: dict
    object_mask: np.ndarray = field(repr=False)
    observed_mask: np.ndarray = field(repr=False)
    near_mask: np.ndarray = field(repr=False)
    uncertain_border_mask: np.ndarray = field(repr=False)
    distance_px: np.ndarray = field(repr=False)

    @property
    def kept(self):
        return [r for r in self.records if r['status']=='kept']

    @property
    def excluded(self):
        return [r for r in self.records if r['status']=='excluded']

    @property
    def pending(self):
        return [r for r in self.records if r['status']=='pending']

    @property
    def ignored(self):
        return [r for r in self.records if r['status']=='ignored']

    def to_dict(self):
        return _json_value(dict(records=self.records,parameters=self.parameters,summary=self.summary))

    def render_debug(self,color=None,*,color_order='RGB',depth=None,depth_scale_m=.001,
                     show_ids=False,panel=True):
        """Before/after graph overlays; pending is not silently removed."""
        cv=_opencv();shape=self.object_mask.shape
        if color_order not in ('RGB','BGR') or type(show_ids) is not bool or type(panel) is not bool:
            raise ValueError('RGB/BGR and boolean display switches required')
        if color is not None:
            a=np.asarray(color)
            if a.dtype!=np.uint8 or a.shape!=(*shape,3):raise ValueError('color must be uint8 HxWx3 on the result grid')
            base=(a if color_order=='RGB' else a[...,::-1]).copy()
        elif depth is not None:
            base=depth_canvas(depth,depth_scale_m)
            if base.shape[:2]!=shape:raise ValueError('depth/result shape mismatch')
        else:base=np.repeat(np.where(self.object_mask,75,25).astype(np.uint8)[...,None],3,axis=2)
        before=base.copy();after=base.copy()
        after[self.near_mask]=(0.7*after[self.near_mask]+0.3*np.array([235,100,100])).astype(np.uint8)
        after[self.uncertain_border_mask]=(210,65,210)
        palette=dict(kept=(30,230,80),excluded=(250,80,65),pending=(255,210,40),ignored=(145,145,145))
        for r in self.records:
            try:q=np.asarray(r['segment'].get('points_xy',[]),float)
            except (ValueError,TypeError):continue
            if q.ndim!=2 or q.shape[1]!=2:continue
            good=np.isfinite(q).all(axis=1)&(q==np.rint(q)).all(axis=1)
            good &= (q[:,0]>=0)&(q[:,0]<shape[1])&(q[:,1]>=0)&(q[:,1]<shape[0])
            for i in range(len(q)):
                if not good[i]:continue
                xy=tuple(q[i].astype(int));ink=palette[r['status']]
                cv.circle(before,xy,1,(225,225,225),-1);cv.circle(after,xy,1,ink,-1)
                if i and good[i-1]:
                    cv.line(before,tuple(q[i-1].astype(int)),xy,(225,225,225),1)
                    cv.line(after,tuple(q[i-1].astype(int)),xy,ink,1)
            if show_ids and good.any():
                xy=tuple(q[np.flatnonzero(good)[len(np.flatnonzero(good))//2]].astype(int))
                cv.putText(after,f"{r['channel']}:{r['segment_id']}",xy,cv.FONT_HERSHEY_SIMPLEX,.32,palette[r['status']],1)
        h,w=shape;canvas=np.full((h+30,max(w,360)*2,3),24,np.uint8);cell=max(w,360)
        canvas[30:,:w]=before;canvas[30:,cell:cell+w]=after
        for text,x in [('Original graph (all recorded paths)',5),('Kept / excluded / pending / ignored',cell+5)]:cv.putText(canvas,text,(x,20),cv.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
        if panel:
            footer=np.full((90,canvas.shape[1],3),24,np.uint8)
            lines=[f"near <= {self.parameters['near_px']} px; exclude fraction >= {self.parameters['reject_fraction']}; holes={self.parameters['include_holes']}",
                   f"Kept {len(self.kept)}, excluded {len(self.excluded)}, pending {len(self.pending)}, ignored {len(self.ignored)}",
                   'Statistics: unique strong/weak points; inferred points excluded.',
                   'Magenta: frame/known-domain uncertainty. No physical boundary guarantee.']
            for i,line in enumerate(lines):cv.putText(footer,line,(5,18+20*i),cv.FONT_HERSHEY_SIMPLEX,.38,(235,235,235),1)
            canvas=np.concatenate([canvas,footer])
        return canvas if color_order=='RGB' else np.ascontiguousarray(canvas[...,::-1])


def filter_outer_proximity(result,*,near_px=8.,reject_fraction=.7,include_holes=False,
                           observed_mask=None):
    """Select graph candidates by unique observed sample fraction, not minimum distance."""
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in (near_px,reject_fraction)):
        raise ValueError('numeric, non-boolean distance and fraction required')
    if not np.isscalar(near_px) or not np.isscalar(reject_fraction) or not np.isfinite([near_px,reject_fraction]).all() or near_px<0 or not 0<reject_fraction<=1:
        raise ValueError('near_px must be finite nonnegative; reject_fraction finite in (0,1]')
    if type(include_holes) is not bool:raise ValueError('include_holes must be bool')
    cv=_opencv();obj=np.asarray(result.object_mask)
    if obj.dtype!=bool or obj.ndim!=2 or min(obj.shape)<2:raise ValueError('result object_mask must be bool HxW')
    observed=np.ones_like(obj) if observed_mask is None else bool_mask(observed_mask,obj.shape,'observed_mask').copy()
    visible=obj&observed;h,w=obj.shape
    _,labels,stats,_=cv.connectedComponentsWithStats(visible.astype(np.uint8),8)
    distance=np.full(obj.shape,np.nan,dtype=np.float32)
    # Fill holes per component, not globally: nested separate islands retain their own exterior.
    for label in range(1,len(stats)):
        x,y,bw,bh,_=stats[label];member=labels[y:y+bh,x:x+bw]==label;domain=member.astype(np.uint8)
        if not include_holes:
            contours,_=cv.findContours(domain,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
            domain=np.zeros_like(domain);cv.drawContours(domain,contours,-1,1,cv.FILLED)
        d=cv.distanceTransform(np.pad(domain,1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
        tile=distance[y:y+bh,x:x+bw];tile[member]=d[member]
    # Mark only object contact with the frame or an explicitly unobserved region.
    safe=cv.erode(observed.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv.BORDER_CONSTANT,borderValue=0)>0
    artificial=visible&~safe
    border_distance=cv.distanceTransform((~artificial).astype(np.uint8),cv.DIST_L2,cv.DIST_MASK_PRECISE) if artificial.any() else np.full(obj.shape,np.inf)
    uncertain=visible&(border_distance<=near_px)
    records=[];channels={}
    for channel in ('color','depth'):
        graph=result.topology.get(channel)
        channels[channel]='not_available' if graph is None else graph.get('status','unknown')
        for source in ([] if graph is None else graph.get('segments',[])):
            row=dict(channel=channel,segment_id=source.get('id'),status='pending',source_status=source.get('status'),
                     segment=deepcopy(source),observed_points_xy=[],distances_px=[],observed_count=0,
                     inferred_unique_count=0,near_observed_count=None,invalid_input_point_count=None,near_fraction=None,min_distance_px=None,uncertain_border_count=0,reasons=[])
            records.append(row)
            if source.get('status')!='candidate':
                row.update(status='ignored',reasons=['source_status_is_not_candidate']);continue
            if graph.get('status')!='computed':row['reasons'].append('topology_not_computed')
            try:q=np.asarray(source.get('points_xy'),float);e=np.asarray(source.get('evidence'))
            except (TypeError,ValueError):row['reasons'].append('invalid_points_or_evidence');continue
            if q.ndim!=2 or q.shape[1]!=2 or e.ndim!=1 or len(e)!=len(q) or not np.isin(e,['strong','weak','inferred']).all():
                row['reasons'].append('invalid_points_or_evidence');continue
            finite_integer=np.isfinite(q).all(axis=1)&(q==np.rint(q)).all(axis=1)
            bounded=(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h)
            row['invalid_input_point_count']=int((~(finite_integer&bounded)).sum())
            if not finite_integer.all():row['reasons'].append('nonfinite_or_noninteger_points');continue
            if not bounded.all():row['reasons'].append('out_of_range_points');continue
            q=q.astype(int);points=np.unique(q[e!='inferred'],axis=0)
            if not visible[q[:,1],q[:,0]].all():row['reasons'].append('path_crosses_outside_object_or_observed_domain')
            row['inferred_unique_count']=len(np.unique(q[e=='inferred'],axis=0))
            row['observed_points_xy']=points.tolist();row['observed_count']=len(points)
            if not len(points):row['reasons'].append('no_observed_points');continue
            x,y=points.T
            if not visible[y,x].all():row['reasons'].append('observed_points_outside_object_or_observed_domain');continue
            ids=np.unique(labels[y,x]);row['object_component_ids']=ids.tolist()
            if len(ids)!=1:row['reasons'].append('multiple_object_components')
            d=distance[y,x];row.update(distances_px=d.astype(float).tolist(),min_distance_px=float(d.min()),near_observed_count=int((d<=near_px).sum()),near_fraction=float(np.mean(d<=near_px)),uncertain_border_count=int(uncertain[y,x].sum()))
            if row['uncertain_border_count']:row['reasons'].append('near_frame_or_known_unobserved_boundary')
            if row['reasons']:continue
            row['status']='excluded' if row['near_fraction']>=reject_fraction else 'kept'
            row['reasons']=['observed_near_fraction_at_or_above_limit' if row['status']=='excluded' else 'observed_near_fraction_below_limit']
    counts={s:sum(r['status']==s for r in records) for s in ('kept','excluded','pending','ignored')}
    summary=dict(status='computed' if records else 'no_segments',channels=channels,counts=counts,artificial_border_pixels=int(artificial.sum()),
                 observation_domain_supplied=observed_mask is not None,
                 warnings=['mask_geometry_not_physical_boundary_identity','undisclosed_internal_ROI_cut_cannot_be_detected','inferred_points_are_not_observed_support','point_fraction_is_not_arclength_fraction'])
    return BoundaryProximityResult(records,dict(near_px=float(near_px),reject_fraction=float(reject_fraction),include_holes=include_holes,
        distance_definition='euclidean_pixel_center_to_nearest_zero_of_per_component_hole_filled_or_original_mask',
        sample_definition='unique_strong_or_weak_xy',frame_policy='pending_if_any_observed_point_near_known_artificial_border'),
        summary,obj.copy(),observed.copy(),visible&(distance<=near_px),uncertain.copy(),distance)
