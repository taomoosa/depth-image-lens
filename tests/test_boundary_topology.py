"""Topology contracts on analytic evidence, not physical boundary accuracy."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from depth_image_lens import OverheadBoundaryDetector
from depth_image_lens._boundary_topology import trace_evidence


def trace(mask,weak=None,domain=None,**kw):
    p=OverheadBoundaryDetector(trace_boundaries=True,**kw).parameters
    return trace_evidence(mask,mask if weak is None else weak,np.ones_like(mask) if domain is None else domain,p)


def line_scene():
    color=np.full((90,130,3),190,np.uint8);mask=np.zeros((90,130),bool);mask[5:-5,5:-5]=True
    color[44:47,20:110]=30;depth=np.full(mask.shape,1500,np.uint16)
    return color,depth,mask


class TopologyTests(unittest.TestCase):
    def test_straight_and_deterministic(self):
        m=np.zeros((70,90),bool);m[30,10:75]=True;a,_=trace(m);b,_=trace(m)
        self.assertEqual(a,b);self.assertEqual(a['counts']['segments'],1);self.assertEqual(a['counts']['corners'],0)
        self.assertEqual(sum(n['kind']=='endpoint' for n in a['nodes']),2)
        self.assertEqual(a['segments'][0]['points_xy'][0],[10,30]);json.dumps(a,allow_nan=False)

    def test_weak_connected_not_isolated(self):
        weak=np.zeros((70,90),bool);weak[30,10:75]=1;weak[50,10:30]=1;strong=weak.copy();strong[30,35:50]=0;strong[50]=0
        a,_=trace(strong);b,maps=trace(strong,weak)
        self.assertEqual(a['counts']['components'],2);self.assertEqual(b['counts']['components'],1)
        self.assertFalse(maps['retained_mask'][50].any());self.assertIn('weak',b['segments'][0]['evidence']);self.assertEqual(b['counts']['inferred_pixels'],0)

    def test_gap_control_and_provenance(self):
        m=np.zeros((70,90),bool);m[30,10:40]=1;m[30,42:75]=1
        a,_=trace(m);b,maps=trace(m,max_gap_px=2)
        self.assertEqual(a['counts']['components'],2);self.assertEqual(b['counts']['components'],1)
        self.assertEqual(b['counts']['inferred_pixels'],2);self.assertEqual(maps['inferred_mask'].sum(),2)
        self.assertEqual(b['segments'][0]['evidence'].count('inferred'),2)

    def test_large_gap_and_missing_domain_not_connected(self):
        m=np.zeros((70,90),bool);m[30,10:40]=1;m[30,45:75]=1;d=np.ones_like(m);d[:,42]=0
        for gap,domain in [(2,None),(8,d)]:
            g,_=trace(m,max_gap_px=gap,domain=domain);self.assertEqual(g['counts']['components'],2);self.assertEqual(g['counts']['inferred_pixels'],0)

    def test_parallel_and_ambiguous_endpoints(self):
        m=np.zeros((80,110),bool);m[30,10:50]=1;m[34,10:50]=1
        g,_=trace(m,max_gap_px=8);self.assertEqual(g['counts']['components'],2);self.assertEqual(g['counts']['inferred_pixels'],0)
        m[:]=0;m[40,10:50]=1;m[39,55:95]=1;m[42,55:95]=1
        g,_=trace(m,max_gap_px=8,gap_angle_deg=45);self.assertEqual(g['counts']['inferred_pixels'],0)

    def test_crossing_shortcut_rejected(self):
        m=np.zeros((80,110),bool);m[40,10:50]=1;m[40,56:95]=1;m[15:65,53]=1
        g,_=trace(m,max_gap_px=8);self.assertEqual(g['counts']['inferred_pixels'],0)

    def test_L_corner_threshold(self):
        m=np.zeros((70,90),bool);m[15:50,20]=1;m[49,20:65]=1
        sharp,_=trace(m,corner_turn_deg=45);smooth,_=trace(m,corner_turn_deg=110)
        self.assertEqual(sharp['counts']['corners'],1);self.assertEqual(sharp['counts']['segments'],2)
        self.assertEqual(smooth['counts']['corners'],0);self.assertEqual(smooth['counts']['segments'],1)
        self.assertAlmostEqual(next(n['turn_deg'] for n in sharp['nodes'] if n['kind']=='corner'),90)

    def test_curve_not_forced_straight(self):
        import cv2
        m=np.zeros((140,160),np.uint8);cv2.ellipse(m,(80,80),(55,55),0,190,350,1,1)
        g,_=trace(m.astype(bool),corner_turn_deg=60,corner_window_px=6)
        self.assertEqual(g['counts']['corners'],0);self.assertEqual(g['counts']['components'],1)
        self.assertGreater(g['segments'][0]['length_px'],100)

    def test_T_X_junction_independent_of_corner_threshold(self):
        for kind in ['T','X']:
            m=np.zeros((80,90),bool)
            if kind=='T':m[20,15:75]=1;m[20:65,45]=1
            else:
                for t in range(15,66):m[t,t]=1;m[t,80-t]=1
            a,_=trace(m,corner_turn_deg=30);b,_=trace(m,corner_turn_deg=150)
            for g in (a,b):
                js=[n for n in g['nodes'] if n['kind']=='junction'];self.assertEqual(len(js),1)
                self.assertEqual(len(js[0]['segment_ids']),3 if kind=='T' else 4);self.assertTrue(js[0]['branch_angles_deg'])

    def test_loop_preserved_and_split_at_corners(self):
        m=np.zeros((70,90),bool);m[15,15:65]=1;m[50,15:65]=1;m[15:51,15]=1;m[15:51,64]=1
        a,_=trace(m,corner_turn_deg=110);b,_=trace(m,corner_turn_deg=45)
        self.assertEqual(len(a['segments']),1);self.assertTrue(a['segments'][0]['closed'])
        self.assertEqual(a['segments'][0]['points_xy'][0],a['segments'][0]['points_xy'][-1])
        self.assertEqual(b['counts']['corners'],4);self.assertEqual(b['counts']['segments'],4)

    def test_loop_at_junction_has_two_distinct_arms(self):
        m=np.zeros((90,90),bool);m[20,20:61]=1;m[60,20:61]=1
        m[20:61,20]=1;m[20:61,60]=1;m[40,60:80]=1
        g,_=trace(m,corner_turn_deg=150)
        j=next(n for n in g['nodes'] if n['kind']=='junction')
        self.assertEqual(len(j['arms']),3);self.assertEqual(len(j['segment_ids']),2)
        self.assertEqual(sorted(round(a['angle_deg']) for a in j['branch_angles_deg']),[90,90,180])

    def test_thick_X_optional_group_preserves_raw_graph(self):
        import cv2,copy
        c=np.full((144,192,3),210,np.uint8);m=np.zeros(c.shape[:2],bool);m[12:-12,12:-12]=1
        paint=np.zeros(m.shape,np.uint8)
        cv2.line(paint,(40,30),(152,114),1,1);cv2.line(paint,(40,114),(152,30),1,1)
        c[cv2.dilate(paint,np.ones((3,3),np.uint8))>0]=70
        kw=dict(color_order='RGB',object_mask=m,diagnostics=True)
        a=OverheadBoundaryDetector(trace_boundaries=True).detect(c,**kw)
        b=OverheadBoundaryDetector(trace_boundaries=True,junction_cluster_px=4).detect(c,**kw)
        self.assertEqual(a.topology['color']['counts']['junctions'],2)
        grouped=b.topology['color']['junction_grouping'];self.assertEqual(len(grouped['groups']),1)
        self.assertEqual(grouped['grouped_junction_count'],1)
        self.assertEqual(len(grouped['groups'][0]['external_arms']),4)
        for kind in ('color','depth'):
            x=copy.deepcopy(a.topology[kind]);y=copy.deepcopy(b.topology[kind])
            x.pop('junction_grouping',None);y.pop('junction_grouping',None);self.assertEqual(x,y)
        np.testing.assert_array_equal(a.color_boundary_mask,b.color_boundary_mask)
        np.testing.assert_array_equal(a.diagnostics['color_topology']['observed_centerline_mask'],b.diagnostics['color_topology']['observed_centerline_mask'])
        for params in [dict(junction_cluster_px=3),dict(junction_cluster_px=4,junction_cluster_angle_deg=0),dict(junction_cluster_px=4,junction_cluster_residual_px=.05)]:
            r=OverheadBoundaryDetector(trace_boundaries=True,**params).detect(c,**kw)
            self.assertFalse(r.topology['color']['junction_grouping']['groups'])
        json.dumps(b.to_dict(),allow_nan=False)

    def test_nearby_distinct_T_junctions_not_grouped(self):
        for kind in ('same_side','opposite','far'):
            m=np.zeros((120,150),bool);m[60,15:130]=1;m[20:60,70]=1
            x=80 if kind=='far' else 74
            if kind=='opposite':m[60:101,x]=1
            else:m[20:60,x]=1
            g,_=trace(m,junction_cluster_px=6)
            self.assertEqual(g['counts']['junctions'],2)
            self.assertEqual(g['junction_grouping']['grouped_junction_count'],2)
            self.assertFalse(g['junction_grouping']['groups'])

    def test_junction_grouping_rejects_interpolated_connector(self):
        from depth_image_lens._boundary_topology import _junction_groups
        nodes=[dict(id=1,kind='junction'),dict(id=2,kind='junction')]
        segments=[dict(id=1,start_node=1,end_node=2,length_px=3,evidence=['strong','inferred','strong'])]
        p=OverheadBoundaryDetector(junction_cluster_px=4).parameters
        r=_junction_groups(nodes,segments,p);self.assertFalse(r['groups'])
        self.assertEqual(r['rejections']['interpolated_connector'],1)

    def test_junction_parameters_and_config(self):
        for kw in [dict(junction_cluster_px=21),dict(junction_cluster_px=True),dict(junction_cluster_residual_px=0),dict(junction_cluster_residual_px=float('nan')),dict(junction_cluster_angle_deg=46)]:
            with self.assertRaises(ValueError):OverheadBoundaryDetector(**kw)
        d=OverheadBoundaryDetector(junction_cluster_px=4,junction_cluster_residual_px=.8,junction_cluster_angle_deg=10)
        self.assertEqual(d.parameters,OverheadBoundaryDetector.from_config(d.parameters).parameters)

    def test_empty_isolated_short_and_budget(self):
        m=np.zeros((20,30),bool);a,_=trace(m);self.assertEqual(a['counts']['segments'],0)
        m[10,10]=1;a,_=trace(m);self.assertEqual(a['nodes'][0]['kind'],'isolated')
        m[10,11]=1;a,_=trace(m,min_path_length_px=5);self.assertEqual(a['segments'][0]['status'],'short_path')
        m=np.ones((501,501),bool);a,_=trace(m);self.assertEqual(a['status'],'insufficient');self.assertIn('budget',a['reason'])

    def test_api_preserves_measured_results_and_diag_independence(self):
        c,z,m=line_scene();before=c.copy();zcopy=z.copy()
        kwargs=dict(color_order='RGB',registered_to_depth=True,object_mask=m)
        a=OverheadBoundaryDetector().detect(c,z,**kwargs);d=OverheadBoundaryDetector(trace_boundaries=True,color_weak_ratio=.5,max_gap_px=2)
        b=d.detect(c,z,diagnostics=True,**kwargs);e=d.detect(c,z,**kwargs)
        self.assertEqual(a.metrics,b.metrics);self.assertEqual(a.stages,b.stages);self.assertEqual(b.topology,e.topology)
        for key,value in vars(a).items():
            if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,getattr(b,key))
        self.assertFalse(a.topology);self.assertTrue(b.topology['color']['segments'])
        np.testing.assert_array_equal(c,before);np.testing.assert_array_equal(z,zcopy);json.dumps(b.to_dict(),allow_nan=False)

    def test_missing_or_bad_depth_keeps_color_and_empty_mask(self):
        c,z,m=line_scene()
        for depth in [None,z[:-1],np.zeros_like(z)]:
            r=OverheadBoundaryDetector(trace_boundaries=True).detect(c,depth,color_order='RGB',registered_to_depth=True,object_mask=m)
            self.assertTrue(r.topology['color']['segments']);self.assertFalse(r.topology['depth']['segments'])
        r=OverheadBoundaryDetector(trace_boundaries=True).detect(c,z,color_order='RGB',registered_to_depth=True,object_mask=np.zeros_like(m))
        self.assertFalse(r.topology['color']['segments']);self.assertFalse(r.topology['depth']['segments'])

    def test_depth_step_graph_noise_and_hole(self):
        c,z,m=line_scene();z[:,65:]-=30;z[40:45,62:69]=0
        r=OverheadBoundaryDetector(trace_boundaries=True,max_gap_px=8,depth_weak_ratio=.5).detect(c,z,color_order='RGB',registered_to_depth=True,object_mask=m,diagnostics=True)
        g=r.topology['depth'];self.assertTrue(g['segments']);self.assertFalse(r.diagnostics['depth_topology']['inferred_mask'][z==0].any())
        rng=np.random.default_rng(4);noisy=1500+rng.integers(-1,2,z.shape)
        q=OverheadBoundaryDetector(trace_boundaries=True,depth_weak_ratio=.5).detect(c,noisy,color_order='RGB',registered_to_depth=True,object_mask=m)
        self.assertFalse(q.topology['depth']['segments'])

    def test_debug_RGB_BGR_save_nonmutation(self):
        c,z,m=line_scene();d=OverheadBoundaryDetector(trace_boundaries=True);r=d.detect(c,z,color_order='RGB',registered_to_depth=True,object_mask=m)
        a=r.render_debug(c,topology_labels=True);b=r.render_debug(c[...,::-1],color_order='BGR',topology_labels=True)
        np.testing.assert_array_equal(a,b[...,::-1])
        self.assertTrue(r.render_debug(depth=z).size)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'debug.png';r.save_debug(p,c);self.assertTrue(p.exists())
            with self.assertRaises(FileExistsError):r.save_debug(p,c)

    def test_parameter_validation_and_roundtrip(self):
        for kw in [{'trace_boundaries':1},{'color_weak_ratio':0},{'depth_weak_ratio':1.1},{'max_gap_px':21},{'max_gap_px':1.5},{'gap_angle_deg':90},{'corner_turn_deg':0},{'corner_window_px':1},{'tangent_window_px':51},{'min_path_length_px':-1}]:
            with self.assertRaises(ValueError):OverheadBoundaryDetector(**kw)
        d=OverheadBoundaryDetector(trace_boundaries=True,color_weak_ratio=.5)
        self.assertEqual(d.parameters,OverheadBoundaryDetector.from_config(d.parameters).parameters)


if __name__=='__main__':unittest.main()
