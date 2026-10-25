"""Independent visual grasp hypotheses from separated hand/object regions."""
from dataclasses import dataclass, field, replace
import numpy as np
from .depth import _numeric_image, _depth_16uc1_to_meters
from .geometry import bool_mask, largest_component
from .color import _erode


def _component_labels(points):
    """Four-neighbor labels on sparse observed boundary coordinates."""
    remaining=set(points);labels={};component=0
    for start in sorted(remaining):
        if start not in remaining:continue
        component+=1;remaining.remove(start);pending=[start]
        while pending:
            y,x=pending.pop();labels[(y,x)]=component
            for neighbor in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor);pending.append(neighbor)
    return labels


def _corresponding_regions(pairs):
    """Partition actual matched pairs until both endpoint sets are contiguous.

    Repartition subsets because removing pairs assigned to another endpoint
    component can split a previously connected set. No bridge between hands or
    through the full object mask is required.
    """
    if not pairs:return []
    pending=[pairs];regions=[]
    while pending:
        group=pending.pop()
        hand_labels=_component_labels((hy,hx) for hy,hx,oy,ox in group)
        object_labels=_component_labels((oy,ox) for hy,hx,oy,ox in group)
        subgroups={}
        for pair in group:
            hy,hx,oy,ox=pair;key=(hand_labels[(hy,hx)],object_labels[(oy,ox)])
            subgroups.setdefault(key,[]).append(pair)
        if len(subgroups)>1:
            pending.extend(subgroups.values());continue
        def bbox(points):
            ys,xs=zip(*points)
            return [int(min(xs)),int(min(ys)),int(max(xs)+1),int(max(ys)+1)]
        regions.append({'hand_pixels':len(hand_labels),'object_pixels':len(object_labels),
            'matched_pairs':len(group),'hand_bbox_xyxy':bbox(hand_labels),
            'object_bbox_xyxy':bbox(object_labels)})
    return sorted(regions,key=lambda region:(region['hand_bbox_xyxy'],region['object_bbox_xyxy']))


@dataclass(frozen=True)
class GraspResult:
    sides: dict
    parameters: dict
    scope: str = 'visual_proximity_inference_not_physical_grasp'
    identity: str = 'caller_image_left_right_not_physical_robot_identity'
    segmentation_source: str = 'caller_separated_masks'
    diagnostics: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        """JSON-safe measurements; diagnostic arrays remain available separately."""
        return {'sides':self.sides,'parameters':self.parameters,'scope':self.scope,'identity':self.identity,'segmentation_source':self.segmentation_source}


class IndependentGraspDetector:
    """No bridge, sag, motion or force prerequisite; one side cannot veto the other.

    Hand masks must select end effectors, not entire arms. Registration and mask
    semantics are caller assertions. Pixel proximity with consistent optical-axis
    depth is a visual hypothesis, never verification of physical holding.
    """
    def __init__(self, *, contact_radius_px=3, depth_tolerance_m=.04,
                 min_contact_pixels=3, min_consistent_fraction=.6, min_valid_fraction=.8):
        if type(contact_radius_px) is not int or not 1<=contact_radius_px<=8:
            raise ValueError('integer contact_radius_px in [1,8] required')
        if type(min_contact_pixels) is not int or min_contact_pixels<1:
            raise ValueError('positive integer min_contact_pixels required')
        if (not np.isfinite([depth_tolerance_m,min_consistent_fraction,min_valid_fraction]).all()
                or depth_tolerance_m<=0 or not 0<min_consistent_fraction<=1 or not 0<min_valid_fraction<=1):
            raise ValueError('positive finite depth tolerance and fractions in (0,1] required')
        self.contact_radius_px=contact_radius_px;self.depth_tolerance_m=float(depth_tolerance_m)
        self.min_contact_pixels=min_contact_pixels;self.min_consistent_fraction=float(min_consistent_fraction)
        self.min_valid_fraction=float(min_valid_fraction)

    def detect_16uc1(self, depth, object_mask, hand_masks, *, depth_scale_m=.001, **options):
        return self.detect(_depth_16uc1_to_meters(depth,depth_scale_m=depth_scale_m),
                           object_mask,hand_masks,depth_unit='m',**options)

    def detect(self, depth, object_mask, hand_masks, *, depth_unit, registered_to_depth,
               observed_mask=None, uncertain_mask=None, hand_visibility=('visible','visible'),
               mask_complete=(False,False), diagnostics=False):
        """Same-grid ndarray inputs, explicit m/mm depth and bool separated masks.

        hand_masks has exactly left/right bool arrays, in caller image coordinates.
        mask_complete attests each hand and its relevant object search region for
        negative inference. Partial hands can support a positive local hypothesis,
        but an absent offscreen hand cannot. No missing/ambiguous pixels are filled.
        """
        if registered_to_depth is not True:raise ValueError('explicit registered_to_depth=True required')
        if depth_unit not in ('m','mm'):raise ValueError('explicit m or mm depth_unit required')
        if type(diagnostics) is not bool:raise ValueError('diagnostics must be bool')
        if not isinstance(hand_visibility,(tuple,list)) or len(hand_visibility)!=2 or any(v not in ('visible','partial','hidden') for v in hand_visibility):
            raise ValueError('left/right visibility must be visible, partial or hidden')
        if not isinstance(mask_complete,(tuple,list)) or len(mask_complete)!=2 or any(type(v) is not bool for v in mask_complete):
            raise ValueError('left/right mask_complete must contain two bool values')
        if not isinstance(hand_masks,dict) or set(hand_masks)!= {'left','right'}:
            raise ValueError('explicit left and right hand_masks required')
        z=_numeric_image(depth,'depth')*(.001 if depth_unit=='mm' else 1.)
        obj=bool_mask(object_mask,z.shape,'object_mask')
        hands={k:bool_mask(v,z.shape,k+'_hand_mask') for k,v in hand_masks.items()}
        observed=np.ones(z.shape,bool) if observed_mask is None else bool_mask(observed_mask,z.shape,'observed_mask')
        uncertain=np.zeros(z.shape,bool) if uncertain_mask is None else bool_mask(uncertain_mask,z.shape,'uncertain_mask')
        valid=np.isfinite(z)&(z>0)&observed
        overlap=(obj&(hands['left']|hands['right']))|(hands['left']&hands['right'])
        ambiguous=overlap|uncertain
        object_eligible=obj&~ambiguous;object_boundary=object_eligible&~_erode(object_eligible,1)
        results={};debug={}
        for index,name in enumerate(('left','right')):
            hand=hands[name]&~ambiguous;boundary=hand&~_erode(hand,1)
            warnings=[]
            if not mask_complete[index]:warnings.append('local_mask_completeness_unverified')
            if hand_visibility[index]!='visible':warnings.append('hand_'+hand_visibility[index])
            if (hands[name]&ambiguous).any():warnings.append('ambiguous_hand_pixels_excluded')
            if (hand&~valid).any():warnings.append('hand_depth_unobserved')
            if hand.any() and (hand[0].any() or hand[-1].any() or hand[:,0].any() or hand[:,-1].any()):
                warnings.append('hand_touches_image_boundary')
            hy,hx=np.nonzero(boundary);nearest=np.full(len(hx),np.inf);pairs=[];uncertain_near=False
            # Search only bounded offsets. Keep nearest image-distance ties so a
            # more distant matching surface cannot bypass a closer depth mismatch.
            for dy in range(-self.contact_radius_px,self.contact_radius_px+1):
                for dx in range(-self.contact_radius_px,self.contact_radius_px+1):
                    distance=float(np.hypot(dx,dy))
                    if distance==0 or distance>self.contact_radius_px:continue
                    oy,ox=hy+dy,hx+dx;inside=(oy>=0)&(oy<z.shape[0])&(ox>=0)&(ox<z.shape[1])
                    ids=np.flatnonzero(inside)
                    uncertain_near |= bool(ambiguous[oy[ids],ox[ids]].any())
                    ids=ids[object_boundary[oy[ids],ox[ids]]]
                    if not len(ids):continue
                    nearest[ids]=np.minimum(nearest[ids],distance)
                    pairs.append((distance,ids,oy[ids],ox[ids]))
            near=np.isfinite(nearest);valid_hand=np.zeros(len(hx),bool);consistent_hand=np.zeros(len(hx),bool)
            hand_contact=np.zeros(z.shape,bool);object_contact=np.zeros(z.shape,bool);deltas=[];pair_trace=[]
            nearest_pair_count=0;valid_pair_count=0;matched_pairs=[]
            for distance,ids,oy,ox in pairs:
                choose=distance==nearest[ids];ids,oy,ox=ids[choose],oy[choose],ox[choose]
                nearest_pair_count+=len(ids)
                good=valid[hy[ids],hx[ids]]&valid[oy,ox]
                valid_pair_count+=int(good.sum())
                ids,oy,ox=ids[good],oy[good],ox[good]
                if not len(ids):continue
                valid_hand[ids]=True
                delta=np.abs(z[hy[ids],hx[ids]]-z[oy,ox]);deltas.extend(delta.tolist())
                tolerance=4*np.finfo(float).eps*np.maximum(np.abs(z[hy[ids],hx[ids]]),np.abs(z[oy,ox]))
                matched=delta<=self.depth_tolerance_m+tolerance
                consistent_hand[ids[matched]]=True
                hand_contact[hy[ids[matched]],hx[ids[matched]]]=True;object_contact[oy[matched],ox[matched]]=True
                matched_pairs.extend(zip(hy[ids[matched]].tolist(),hx[ids[matched]].tolist(),
                                         oy[matched].tolist(),ox[matched].tolist()))
                if diagnostics:
                    pair_trace.extend([[int(hx[i]),int(hy[i]),int(x),int(y),float(v),bool(ok)]
                        for i,y,x,v,ok in zip(ids,oy,ox,delta,matched)])
            near_count=int(near.sum());valid_count=int(valid_hand.sum());support=int(consistent_hand.sum())
            coverage=min(valid_count/max(near_count,1),valid_pair_count/max(nearest_pair_count,1));fraction=support/max(valid_count,1)
            hand_component=largest_component(hand_contact);object_component=largest_component(object_contact)
            regions=_corresponding_regions(matched_pairs)
            corresponding_support=any(region['hand_pixels']>=self.min_contact_pixels
                and region['object_pixels']>=self.min_contact_pixels for region in regions)
            positive=corresponding_support and fraction>=self.min_consistent_fraction
            if hand_component>=self.min_contact_pixels and object_component>=self.min_contact_pixels and not corresponding_support:
                warnings.append('support_split_across_contact_regions')
            if near_count and coverage<self.min_valid_fraction:warnings.append('contact_depth_coverage_low')
            if support<valid_count:warnings.append('some_near_depth_evidence_inconsistent')
            if uncertain_near:warnings.append('nearby_segmentation_uncertain')
            absence_allowed=(not uncertain_near and mask_complete[index] and hand_visibility[index]=='visible'
                and 'hand_touches_image_boundary' not in warnings and not (hands[name]&ambiguous).any())
            estimate='unknown';reason='insufficient_local_evidence'
            if not hand.any():reason='no_visible_separated_hand_region'
            elif not object_eligible.any():reason='no_separated_object_region'
            elif positive:
                estimate='likely';reason='supported_boundary_proximity_with_consistent_depth'
            elif near_count==0:
                reason='no_object_boundary_within_radius'
                if absence_allowed:estimate='unlikely'
            elif valid_count and coverage>=self.min_valid_fraction and support==0:
                reason='near_boundaries_depth_separated'
                if absence_allowed:estimate='unlikely'
            elif valid_count:reason='limited_or_mixed_contact_support'
            if hand_visibility[index]=='hidden' and hand.any():
                warnings.append('visibility_declaration_conflicts_with_supplied_pixels')
                estimate='unknown';reason='hidden_hand_pixels_not_verified_visible'
            results[name]={'estimate':estimate,'reason':reason,'physical_grasp':'unknown','warnings':warnings,
                'image_identity':name,'hand_visibility':hand_visibility[index],
                'mask_completeness_attested':mask_complete[index],'stages':{
                    'boundary_proximity':{'status':'computed' if hand.any() and object_eligible.any() else 'insufficient_input',
                        'hand_pixels':int(hand.sum()),'hand_boundary_pixels':len(hx),'near_hand_boundary_pixels':near_count,
                        'minimum_gap_px':float(nearest[near].min()) if near.any() else None,
                        'search_radius_px':self.contact_radius_px},
                    'depth_consistency':{'status':'computed' if valid_count else 'insufficient_input',
                        'valid_near_hand_pixels':valid_count,'valid_near_fraction':coverage,
                        'nearest_pair_count':nearest_pair_count,'valid_nearest_pair_count':valid_pair_count,
                        'consistent_hand_pixels':support,'consistent_object_pixels':int(object_contact.sum()),
                        'consistent_fraction':fraction,'hand_contact_component_pixels':hand_component,
                        'object_contact_component_pixels':object_component,
                        'corresponding_contact_regions':regions,
                        'depth_delta_quantiles_m':[float(x) for x in np.quantile(deltas,[0,.5,.95])] if deltas else None},
                    'grasp_inference':{'status':'inferred' if estimate!='unknown' else 'insufficient_input',
                        'estimate':estimate,'negative_evidence_complete':absence_allowed,
                        'checks':{'hand_component_sufficient':hand_component>=self.min_contact_pixels,
                            'object_component_sufficient':object_component>=self.min_contact_pixels,
                            'corresponding_region_sufficient':corresponding_support,
                            'consistent_fraction_sufficient':fraction>=self.min_consistent_fraction,
                            'depth_coverage_sufficient_for_negative':coverage>=self.min_valid_fraction}}}}
            if diagnostics:
                near_mask=np.zeros(z.shape,bool);near_mask[hy[near],hx[near]]=True
                debug[name]={'hand_boundary_mask':boundary,'near_hand_boundary_mask':near_mask,
                    'consistent_hand_contact_mask':hand_contact,'consistent_object_contact_mask':object_contact,
                    'valid_nearest_pairs_xy_delta_m_consistent':np.asarray(pair_trace,float).reshape(-1,6)}
        parameters={k:getattr(self,k) for k in ('contact_radius_px','depth_tolerance_m','min_contact_pixels','min_consistent_fraction','min_valid_fraction')}
        if diagnostics:debug.update(sensor_invalid_mask=~(np.isfinite(z)&(z>0)),caller_unobserved_mask=~observed,
            ambiguous_label_mask=ambiguous,eligible_object_mask=object_eligible)
        return GraspResult(results,parameters,diagnostics=debug)

    def detect_color_candidates(self, depth, colors, hand_regions, *, depth_unit, **options):
        """Use separated RGB appearance candidates inside caller-assigned hand regions.

        Color distance does not identify end effectors or assign physical robot IDs.
        Missing seed classes remain missing evidence; single-class similarity is
        not silently accepted as hand/object separation.
        """
        if colors.summary.get('registration')!='caller_declared':raise ValueError('registered color candidates required')
        if not isinstance(hand_regions,dict) or set(hand_regions)!={'left','right'}:
            raise ValueError('left/right hand regions required')
        hands={k:colors.hand_mask&bool_mask(v,colors.object_mask.shape,k+'_region') for k,v in hand_regions.items()}
        supplied=options.pop('uncertain_mask',None)
        uncertainty=colors.appearance_uncertain_mask.copy()
        if supplied is not None:uncertainty|=bool_mask(supplied,uncertainty.shape,'uncertain_mask')
        result=self.detect(depth,colors.object_mask,hands,depth_unit=depth_unit,registered_to_depth=True,
                           uncertain_mask=uncertainty,**options)
        return replace(result,segmentation_source='seeded_color_appearance_candidates_not_semantic_truth')
