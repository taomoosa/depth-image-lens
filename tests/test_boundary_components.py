"""Component geometry and bounded edge-simple trails; no physical-length claims."""
import copy
import json
import math
import unittest
import numpy as np
from depth_image_lens import OverheadBoundaryDetector
from depth_image_lens._boundary_components import _solve, measure_graph
from .test_boundary_topology import trace, line_scene


def oracle(edges, terminals):
    """Small independent recursive enumeration; never used by the library."""
    vertices=sorted({v for u,w,_ in edges for v in (u,w)})
    closed=len(terminals)<2
    best=0.
    def visit(start, vertex, used, weights):
        nonlocal best
        if used and ((closed and vertex==start) or (not closed and vertex in terminals and vertex!=start)):
            best=max(best,math.fsum(weights))
        for i,(u,v,w) in enumerate(edges):
            if i in used:continue
            if vertex==u:visit(start,v,used|{i},weights+[w])
            elif vertex==v:visit(start,u,used|{i},weights+[w])
    for start in (vertices if closed else terminals):visit(start,start,set(),[])
    return best


def abstract_graph(edges, cid=1, offset=0):
    """Compressed weighted graph fixture; geometry tested independently below."""
    vertices=sorted({v for edge in edges for v in edge[:2]})
    degree={v:0 for v in vertices}
    for u,v,_ in edges:degree[u]+=1;degree[v]+=1
    nodes=[dict(id=v+offset,component=cid,kind='endpoint' if degree[v]==1 else 'corner' if degree[v]==2 else 'junction',
                xy=[10+10*v,10],pixels_xy=[[10+10*v,10]]) for v in vertices]
    segments=[dict(id=i+offset,component=cid,start_node=u+offset,end_node=v+offset,closed=False,
                   points_xy=[[10+10*u,10],[10+10*v,10]],evidence=['strong','strong'],length_px=float(w),status='candidate')
              for i,(u,v,w) in enumerate(edges)]
    return dict(status='computed',nodes=nodes,segments=segments,domain='synthetic_graph')


def measured(mask, **kwargs):
    graph,_=trace(mask,**kwargs)
    return measure_graph(graph,mask.shape,'color',100000,24)


class BoundaryComponentTests(unittest.TestCase):
    def test_straight_sizes_and_endpoints(self):
        mask=np.zeros((70,90),bool);mask[30,10:75]=True
        c=measured(mask)['components'][0]
        self.assertEqual(c['bbox_xyxy'],[10,30,74,30])
        self.assertEqual(c['span_xy_px'],[64,0])
        self.assertEqual(c['bbox_size_px'],[65,1])
        self.assertEqual(c['max_endpoint_trail_length_px'],64)
        self.assertEqual(len(c['endpoint_node_ids']),2)
        self.assertFalse(c['trail']['closed'])

    def test_L_passes_corners_without_branch_sum(self):
        mask=np.zeros((70,90),bool);mask[15:50,20]=True;mask[49,20:65]=True
        c=measured(mask)['components'][0]
        self.assertEqual(len(c['segment_ids']),2)
        self.assertEqual(len(c['trail']['parts']),2)
        self.assertEqual(c['span_xy_px'],[44,34])
        self.assertEqual(c['max_trail_length_px'],78)

    def test_T_Y_trees_do_not_add_all_arms(self):
        for name in ('T','Y'):
            mask=np.zeros((80,90),bool)
            if name=='T':mask[20,15:75]=True;mask[20:65,45]=True
            else:
                mask[40:65,45]=True
                for i in range(21):mask[40-i,45-i]=True;mask[40-i,45+i]=True
            c=measured(mask)['components'][0]
            self.assertTrue(c['exact'])
            self.assertEqual(c['reason'],'tree_terminal_diameter')
            self.assertLess(c['max_trail_length_px'],c['total_segment_length_px'])
            if name=='T':self.assertEqual(c['max_trail_length_px'],74)

    def test_pure_loop_with_and_without_corner_nodes(self):
        mask=np.zeros((70,90),bool);mask[15,15:65]=True;mask[50,15:65]=True
        mask[15:51,15]=True;mask[15:51,64]=True
        for turn in (45,110):
            c=measured(mask,corner_turn_deg=turn)['components'][0]
            self.assertEqual(c['path_kind'],'closed_trail')
            self.assertTrue(c['trail']['closed'])
            self.assertEqual(c['trail']['start_xy'],c['trail']['end_xy'])
            self.assertEqual(c['max_closed_trail_length_px'],168)
            self.assertIsNone(c['max_endpoint_trail_length_px'])
            self.assertEqual(c['total_segment_length_px'],168)

    def test_loop_tail_two_tails_theta_and_barbell(self):
        cases=[([(0,0,8),(0,1,3)],[1],8),
               ([(0,0,8),(0,1,3),(0,2,4)],[1,2],15),
               ([(0,1,2),(0,1,3),(0,1,4)],[],7),
               ([(0,0,8),(0,1,3),(1,1,9)],[],9),
               ([(0,0,8),(0,1,3),(1,1,9),(0,2,5),(1,3,6)],[2,3],31)]
        for edges,terminals,expected in cases:
            vertices=sorted({v for e in edges for v in e[:2]})
            r=_solve(edges,vertices,terminals,[100000],24)
            self.assertTrue(r['exact'])
            self.assertEqual(math.fsum(edges[i][2] for i,_ in r['path']),expected)
            self.assertEqual(len({i for i,_ in r['path']}),len(r['path']))
            vertex=r['start']
            for i,reverse in r['path']:
                u,v,_=edges[i]
                if reverse:u,v=v,u
                self.assertEqual(vertex,u);vertex=v
            self.assertTrue(vertex==r['start'] if len(terminals)<2 else vertex in terminals and vertex!=r['start'])

    def test_small_random_graphs_against_exhaustive_oracle(self):
        rng=np.random.default_rng(117)
        for _ in range(50):
            n=int(rng.integers(2,6));edges=[]
            for v in range(1,n):edges.append((int(rng.integers(v)),v,float(rng.integers(1,10))))
            for _ in range(int(rng.integers(0,4))):edges.append((int(rng.integers(n)),int(rng.integers(n)),float(rng.integers(1,10))))
            degree=np.zeros(n,int)
            for u,v,_ in edges:degree[u]+=1;degree[v]+=1
            terminals=np.flatnonzero(degree==1).tolist()
            r=_solve(edges,list(range(n)),terminals,[100000],24)
            self.assertTrue(r['exact'])
            self.assertEqual(math.fsum(edges[i][2] for i,_ in r['path']),oracle(edges,terminals))

    def test_complexity_caps_never_label_a_lower_bound_maximum(self):
        graph=abstract_graph([(0,1,2),(0,1,3),(0,1,4)])
        for states,edge_limit in [(0,24),(1,24),(100000,2)]:
            r=measure_graph(graph,(100,100),'color',states,edge_limit)
            c=r['components'][0]
            self.assertFalse(c['exact']);self.assertEqual(c['status'],'bounded')
            self.assertIsNone(c['max_trail_length_px'])
            self.assertLessEqual(c['lower_bound_length_px'],7)
            self.assertGreaterEqual(c['upper_bound_length_px'],7)
            self.assertTrue(c['trail']['parts'])
            self.assertLessEqual(r['summary']['search_states'],states)

    def test_search_budget_shared_across_components(self):
        a=abstract_graph([(0,1,2),(0,1,3),(0,1,4)],1)
        b=abstract_graph([(0,1,2),(0,1,3),(0,1,4)],2,10)
        a['nodes']+=b['nodes'];a['segments']+=b['segments']
        r=measure_graph(a,(100,100),'color',16,24)
        self.assertEqual(r['summary']['search_states'],16)
        self.assertTrue(r['components'][0]['exact'])
        self.assertFalse(r['components'][1]['exact'])

    def test_fast_cases_bypass_exponential_search_limits(self):
        edges=[(0,v,float(v)) for v in range(1,1001)]
        r=_solve(edges,list(range(1001)),list(range(1,1001)),[0],0)
        self.assertTrue(r['exact']);self.assertEqual(r['states'],0)
        self.assertEqual(sum(edges[i][2] for i,_ in r['path']),1999)
        r=_solve([(0,0,3),(0,0,4)], [0], [], [0], 0)
        self.assertTrue(r['exact']);self.assertEqual(len(r['path']),2)

    def test_zero_length_ties_and_no_closed_trail(self):
        edges=[(0,1,0.),(0,2,0.),(0,3,0.)]
        r=_solve(edges,[0,1,2,3],[1,2,3],[0],0)
        self.assertTrue(r['exact']);self.assertEqual(len(r['path']),2)
        graph=abstract_graph([(0,1,3)])
        # A compressed junction can have only one outgoing edge; it is not a
        # terminal endpoint, so do not invent a second endpoint for this query.
        graph['nodes'][0]['kind']='junction'
        c=measure_graph(graph,(100,100),'color',1000,24)['components'][0]
        self.assertEqual(c['status'],'no_closed_trail');self.assertTrue(c['exact'])
        self.assertIsNone(c['max_trail_length_px']);self.assertEqual(c['trail']['parts'],[])

    def test_isolated_and_short_components(self):
        mask=np.zeros((30,40),bool);mask[10,10]=True;mask[20,20:22]=True
        components=measured(mask)['components']
        self.assertEqual(len(components),2)
        isolated=next(c for c in components if not c['segment_ids'])
        self.assertEqual(isolated['bbox_size_px'],[1,1]);self.assertEqual(isolated['span_xy_px'],[0,0])
        self.assertEqual(isolated['status'],'no_edges');self.assertEqual(isolated['max_trail_length_px'],0)
        short=next(c for c in components if c['segment_ids'])
        self.assertEqual(short['max_trail_length_px'],1)  # short_path is retained.

    def test_junction_pixels_in_bbox_and_unmeasured_joins(self):
        graph=abstract_graph([(0,1,4),(0,2,5),(0,3,6)])
        graph['nodes'][0]['pixels_xy']=[[9,8],[10,10],[12,12]]
        graph['segments'][1]['points_xy'][0]=[9,8]
        graph['segments'][2]['points_xy'][0]=[12,12]
        r=measure_graph(graph,(100,100),'color',1000,24)['components'][0]
        self.assertEqual(r['bbox_xyxy'],[9,8,40,12])
        self.assertEqual(r['max_trail_length_px'],11)
        self.assertEqual(len(r['trail']['junction_transitions']),1)
        self.assertEqual(r['trail']['junction_transitions'][0]['length_included_px'],0)

    def test_inferred_gaps_and_weak_evidence_preserved(self):
        strong=np.zeros((70,90),bool);strong[30,10:40]=True;strong[30,42:75]=True
        graph,_=trace(strong,max_gap_px=2)
        c=measure_graph(graph,strong.shape,'color',1000,24)['components'][0]
        self.assertTrue(c['contains_inferred']);self.assertTrue(c['trail']['uses_inferred'])
        self.assertIn('inferred',c['trail']['parts'][0]['evidence'])
        weak=strong.copy();weak[30,40:42]=True
        graph,_=trace(strong,weak=weak)
        c=measure_graph(graph,strong.shape,'color',1000,24)['components'][0]
        self.assertTrue(c['contains_weak']);self.assertTrue(c['trail']['uses_weak'])

    def test_missing_domain_splits_components_and_frame_contact_is_not_certainty(self):
        mask=np.zeros((40,70),bool);mask[20,:]=True;domain=np.ones_like(mask);domain[:,35]=False
        graph,_=trace(mask,domain=domain,max_gap_px=10)
        r=measure_graph(graph,mask.shape,'depth',1000,24)
        self.assertEqual(len(r['components']),2)
        self.assertTrue(all(c['touches_image_boundary'] for c in r['components']))
        self.assertIn('terminals_can_be_caused_by_threshold_roi_crop_or_missing_observations',r['warnings'])

    def test_public_api_color_depth_and_no_mutation(self):
        color,depth,mask=line_scene();depth[:,65:]-=30
        r=OverheadBoundaryDetector(trace_boundaries=True).detect(color,depth,color_order='RGB',
            registered_to_depth=True,object_mask=mask)
        original=copy.deepcopy(r.topology)
        for kind in ('color','depth'):
            report=r.measure_components(kind)
            self.assertEqual(report['status'],'computed');self.assertTrue(report['components'])
            json.dumps(report,allow_nan=False)
            report['components'][0]['trail']['parts'][0]['points_xy'][0][0]=-100
        self.assertEqual(r.topology,original)

    def test_disabled_insufficient_empty_and_argument_validation(self):
        color,depth,mask=line_scene()
        r=OverheadBoundaryDetector().detect(color,color_order='RGB',object_mask=mask)
        self.assertEqual(r.measure_components()['reason'],'trace_boundaries_required')
        for kwargs in ({'kind':'object'},{'max_search_states':True},{'max_search_states':-1},
                       {'max_search_states':1_000_001},{'max_search_edges':65},{'max_search_edges':1.5}):
            with self.assertRaises(ValueError):r.measure_components(**kwargs)
        r=OverheadBoundaryDetector(trace_boundaries=True).detect(color,color_order='RGB',object_mask=mask)
        self.assertEqual(r.measure_components('depth')['status'],'insufficient')
        r=OverheadBoundaryDetector(trace_boundaries=True).detect(np.full_like(color,190),color_order='RGB',object_mask=mask)
        self.assertEqual(r.measure_components()['components'],[])

    def test_invalid_graph_records_rejected(self):
        original=abstract_graph([(0,1,2),(0,1,3),(0,1,4)])
        for change in ('duplicate','negative','bad_node','missing_evidence','out_of_image'):
            graph=copy.deepcopy(original)
            if change=='duplicate':graph['segments'][1]['id']=graph['segments'][0]['id']
            elif change=='negative':graph['segments'][0]['length_px']=-1
            elif change=='bad_node':graph['segments'][0]['end_node']=999
            elif change=='missing_evidence':graph['segments'][0]['evidence']=[]
            else:graph['segments'][0]['points_xy'][0]=[-1,1]
            with self.assertRaises(ValueError):measure_graph(graph,(100,100),'color',1000,24)
