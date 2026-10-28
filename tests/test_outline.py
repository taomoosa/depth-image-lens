"""External-contour geometry invariants and calibrated synthetic evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from depth_image_lens import OuterContourDetector, CameraIntrinsics


def rectangle():
    m=np.zeros((150,190),bool);m[30:121,35:156]=True;return m


def check_partition(test,result):
    for c in result.contours:
        n=c['edge_count'];covered=np.zeros(n,int);arc=0
        for seg in c['segments']:
            idx=(seg['start_index']+np.arange(seg['edge_count']+1))%n
            covered[idx[:-1]]+=1;arc+=seg['arc_length_px']
            test.assertEqual(seg['points_xy'],np.asarray(c['points_xy'])[idx].tolist())
        test.assertTrue((covered==1).all())
        test.assertAlmostEqual(arc,c['arc_length_px'],places=9)
        test.assertEqual(c['coverage_edge_count'],n)
        segs=c['segments']
        for a,b in zip(segs,segs[1:]+segs[:1]):test.assertEqual(a['end_index'],b['start_index'])
        test.assertEqual([x['id'] for x in c['sequence'] if x['type']=='segment'],list(range(len(segs))))


class OuterContourTests(unittest.TestCase):
    def test_rectangle_order_and_length(self):
        r=OuterContourDetector().detect(rectangle());c=r.contours[0];check_partition(self,r)
        self.assertEqual([s['kind'] for s in c['segments']],['straight']*4)
        self.assertEqual(len(c['corners']),4)
        self.assertEqual(c['points_xy'][0],[35,30])
        self.assertEqual(c['orientation'],'clockwise_in_image_coordinates')
        self.assertEqual(sorted(round(s['fit_length_px'],6) for s in c['segments']),[90.,90.,120.,120.])
        self.assertEqual(r.query_edge_length(120,tolerance=.01)['candidate_count'],2)
        self.assertEqual(r.query_edge_length(110,mode='at_least')['candidate_count'],2)
        self.assertEqual(r.query_edge_length(200)['status'],'no_matching_visible_interval')

    def test_rotation_and_translation(self):
        m=np.zeros((220,250),np.uint8);cv2.fillPoly(m,[np.array([[60,30],[210,90],[170,190],[20,130]])],1)
        r=OuterContourDetector().detect(m>0);check_partition(self,r)
        self.assertEqual(len(r.contours[0]['corners']),4)
        shifted=np.pad(m,((10,0),(7,0)))>0;s=OuterContourDetector().detect(shifted)
        np.testing.assert_allclose(np.array(s.contours[0]['points_xy'])-[7,10],r.contours[0]['points_xy'])
        np.testing.assert_allclose([x['fit_length_px'] for x in s.contours[0]['segments']],[x['fit_length_px'] for x in r.contours[0]['segments']])

    def test_smooth_circle_one_closed_curve(self):
        y,x=np.indices((160,160));r=OuterContourDetector().detect((x-80)**2+(y-80)**2<=55**2)
        check_partition(self,r);c=r.contours[0]
        self.assertEqual(len(c['corners']),0);self.assertEqual(len(c['segments']),1)
        self.assertEqual(c['segments'][0]['kind'],'curve');self.assertEqual(c['segments'][0]['start_index'],c['segments'][0]['end_index'])

    def test_rounded_rectangle_tangent_joins(self):
        y,x=np.indices((160,220));dx=np.maximum(abs(x-110)-55,0);dy=np.maximum(abs(y-80)-25,0)
        r=OuterContourDetector().detect(dx*dx+dy*dy<=20**2);check_partition(self,r)
        kinds=[s['kind'] for s in r.contours[0]['segments']]
        self.assertEqual(kinds.count('straight'),4);self.assertEqual(kinds.count('curve'),4)
        self.assertEqual(r.contours[0]['corners'],[])

    def test_corner_threshold_changes_events_not_coverage(self):
        a=OuterContourDetector().detect(rectangle());b=OuterContourDetector(corner_turn_deg=120).detect(rectangle())
        self.assertEqual(len(a.contours[0]['corners']),4);self.assertEqual(len(b.contours[0]['corners']),0)
        check_partition(self,b);self.assertEqual(a.contours[0]['points_xy'],b.contours[0]['points_xy'])

    def test_hole_and_island_are_separate(self):
        m=rectangle();m[50:100,60:130]=False;m[65:85,80:110]=True
        r=OuterContourDetector().detect(m);check_partition(self,r)
        self.assertEqual(len(r.contours),2);self.assertEqual(len(r.holes),1)
        self.assertEqual(r.holes[0]['role'],'hole_not_external_edge')

    def test_disconnected_components(self):
        m=rectangle();m[:,85:100]=False;r=OuterContourDetector().detect(m)
        self.assertEqual(len(r.contours),2);check_partition(self,r)
        self.assertEqual(r.query_edge_length(120,tolerance=1)['candidate_count'],0)

    def test_frame_boundary_not_exposed(self):
        m=np.ones((80,100),bool);r=OuterContourDetector().detect(m);check_partition(self,r)
        self.assertTrue(r.contours[0]['clipped']);self.assertEqual(r.query_edge_length(99,tolerance=1)['status'],'unknown')
        self.assertTrue(all(s['visibility']=='uncertain' for s in r.contours[0]['segments']))

    def test_roi_provenance(self):
        m=rectangle();m[:,110:]=False;obs=np.ones(m.shape,bool);obs[:,110:]=False
        without=OuterContourDetector().detect(m);r=OuterContourDetector().detect(m,observed_mask=obs)
        self.assertFalse(without.contours[0]['clipped']);self.assertTrue(r.contours[0]['clipped']);check_partition(self,r)
        right=[s for s in r.contours[0]['segments'] if all(p[0]==109 for p in s['points_xy'])]
        self.assertTrue(right);self.assertTrue(all(s['visibility']=='uncertain' for s in right))

    def test_empty_and_tiny(self):
        m=np.zeros((20,20),bool);r=OuterContourDetector().detect(m)
        self.assertEqual(r.summary['status'],'no_visible_contour');self.assertEqual(r.query_edge_length(20)['status'],'unknown')
        m[5,5]=True;r=OuterContourDetector().detect(m);check_partition(self,r)
        self.assertEqual(r.contours[0]['status'],'insufficient_support')
        self.assertEqual(r.contours[0]['orientation'],'degenerate_zero_area')

    def test_self_touch_unknown(self):
        m=np.zeros((100,100),bool);m[10:41,10:41]=True;m[41:72,41:72]=True
        r=OuterContourDetector().detect(m);check_partition(self,r)
        self.assertTrue(r.contours[0]['self_touch']);self.assertEqual(r.query_edge_length(30)['status'],'unknown')

    def test_metric_frontoplane_and_units(self):
        m=rectangle();k=CameraIntrinsics(400,400,90,75)
        a=OuterContourDetector().detect(m,depth=np.full(m.shape,1000,np.uint16),intrinsics=k,registered_to_mask=True)
        b=OuterContourDetector().detect(m,depth=np.ones(m.shape),depth_scale_m=1.,intrinsics=k,registered_to_mask=True)
        for x,y in zip(a.contours[0]['segments'],b.contours[0]['segments']):
            self.assertEqual(x['metric']['status'],'measured');self.assertEqual(x['metric'],y['metric'])
            self.assertAlmostEqual(x['metric']['fit_length_m'],x['fit_length_px']/400,places=9)
        self.assertEqual(a.query_edge_length(.3,unit='m',tolerance=.0001)['candidate_count'],2)

    def test_metric_curve_not_confident_straight_edge(self):
        m=rectangle();_,x=np.indices(m.shape);z=1+.04*np.sin(np.pi*(x-35)/120)
        r=OuterContourDetector().detect(m,depth=z,depth_scale_m=1,intrinsics=CameraIntrinsics(400,400,90,75),registered_to_mask=True)
        rows=r.query_edge_length(.3,unit='m',tolerance=.02)['intervals']
        self.assertEqual(sum('nonstraight_3d_interval' in row['reasons'] for row in rows),2)
        self.assertEqual(r.query_edge_length(.3,unit='m',tolerance=.02)['candidate_count'],0)
        self.assertEqual(r.query_edge_length(120,tolerance=.1)['candidate_count'],2)

    def test_unclassified_short_sections_keep_query_unknown(self):
        m=np.zeros((40,40),bool);m[10:21,10:21]=True
        r=OuterContourDetector(corner_window_px=3,min_corner_spacing_px=3).detect(m)
        self.assertTrue(any(s['kind']=='unknown' for s in r.contours[0]['segments']))
        self.assertEqual(r.query_edge_length(10,min_support_px=0)['status'],'unknown')

    def test_query_threshold_boundary(self):
        r=OuterContourDetector().detect(rectangle())
        self.assertEqual(r.query_edge_length(121,tolerance=1)['candidate_count'],2)
        self.assertEqual(r.query_edge_length(121.000001,tolerance=1)['candidate_count'],0)
        m=rectangle();r=OuterContourDetector().detect(m,depth=np.ones(m.shape),depth_scale_m=1,intrinsics=CameraIntrinsics(400,400,90,75),registered_to_mask=True)
        self.assertEqual(r.query_edge_length(.305,unit='m',tolerance=.005)['candidate_count'],2)
        self.assertEqual(r.query_edge_length(.305001,unit='m',tolerance=.005)['candidate_count'],0)
        self.assertEqual(r.query_edge_length(120)['length_kind'],'tls_projected_span')

    def test_default_query_tolerance_is_unit_specific(self):
        m=rectangle();r=OuterContourDetector().detect(m,depth=np.ones(m.shape),depth_scale_m=1,intrinsics=CameraIntrinsics(400,400,90,75),registered_to_mask=True)
        q=r.query_edge_length(.3,unit='m');self.assertEqual(q['tolerance'],.005);self.assertEqual(q['candidate_count'],2)
        self.assertEqual(r.query_edge_length(120)['tolerance'],5.)

    def test_metric_hole_preserves_other_sides(self):
        m=rectangle();z=np.ones(m.shape);z[30,80]=0;k=CameraIntrinsics(400,400,90,75)
        r=OuterContourDetector().detect(m,depth=z,depth_scale_m=1,intrinsics=k,registered_to_mask=True)
        statuses=[s['metric']['status'] for s in r.contours[0]['segments']]
        self.assertEqual(statuses.count('unknown'),1);self.assertEqual(statuses.count('measured'),3)
        bad=next(s for s in r.contours[0]['segments'] if s['metric']['status']=='unknown')
        self.assertIsNone(bad['metric']['arc_length_m']);self.assertFalse(all(bad['metric']['adjacent_pair_valid']))
        self.assertEqual(r.query_edge_length(.3,unit='m',tolerance=.0001)['candidate_count'],1)

    def test_metric_step_not_bridged(self):
        m=rectangle();z=np.ones(m.shape);z[:,95:]=1.1
        r=OuterContourDetector().detect(m,depth=z,depth_scale_m=1,intrinsics=CameraIntrinsics(400,400,90,75),registered_to_mask=True)
        self.assertEqual(sum(s['metric']['status']=='unknown' for s in r.contours[0]['segments']),2)

    def test_bad_optional_metric_does_not_stop_mask(self):
        m=rectangle();baseline=OuterContourDetector().detect(m)
        cases=[dict(depth=np.ones((2,2))),dict(depth=np.ones(m.shape),depth_scale_m=0),dict(depth=np.ones(m.shape),intrinsics=None),dict(depth=np.ones(m.shape),registered_to_mask=False)]
        for opts in cases:
            r=OuterContourDetector().detect(m,**opts)
            self.assertEqual(r.contours[0]['points_xy'],baseline.contours[0]['points_xy'])
            self.assertEqual(len(r.contours[0]['corners']),4);self.assertTrue(r.summary['rejected_optional_metric'])

    def test_all_missing_metric(self):
        m=rectangle();z=np.full(m.shape,np.nan)
        r=OuterContourDetector().detect(m,depth=z,intrinsics=CameraIntrinsics(400,400,90,75),registered_to_mask=True)
        self.assertEqual(r.query_edge_length(.3,unit='m')['status'],'unknown')
        self.assertTrue(all(s['metric']['valid_fraction']==0 for s in r.contours[0]['segments']))
        json.dumps(r.to_dict(),allow_nan=False)

    def test_diagnostic_parity_nonmutation_and_view(self):
        m=rectangle();view=np.repeat(m,2,axis=1)[:,::2];copy=view.copy()
        a=OuterContourDetector().detect(view);b=OuterContourDetector().detect(view,diagnostics=True)
        self.assertEqual(a.to_dict(),b.to_dict());np.testing.assert_array_equal(view,copy)
        self.assertTrue(b.diagnostics);self.assertFalse(a.diagnostics)

    def test_config_replay(self):
        d=OuterContourDetector(smoothing_px=2,line_error_px=2)
        params=json.loads(json.dumps(d.parameters));other=OuterContourDetector(**params)
        self.assertEqual(d.detect(rectangle()).to_dict(),other.detect(rectangle()).to_dict())

    def test_invalid_contracts(self):
        for opts in [dict(smoothing_px=np.nan),dict(corner_turn_deg=0),dict(line_error_px=-1),dict(clip_margin_px=1.2),dict(max_arc_chord_ratio=.9)]:
            with self.assertRaises(ValueError):OuterContourDetector(**opts)
        for mask in [rectangle().astype(np.uint8),np.zeros((5,5,1),bool)]:
            with self.assertRaises(ValueError):OuterContourDetector().detect(mask)
        with self.assertRaises(ValueError):OuterContourDetector().detect(rectangle(),observed_mask=np.ones((2,2),bool))

    def test_query_invalid_and_corner_requirement(self):
        r=OuterContourDetector().detect(rectangle())
        with self.assertRaises(ValueError):r.query_edge_length(0)
        with self.assertRaises(ValueError):r.query_edge_length(20,unit='mm')
        self.assertEqual(r.query_edge_length(120,tolerance=1,require_corner_endpoints=True)['candidate_count'],2)

    def test_debug_save_and_rgb_order(self):
        m=rectangle();color=np.zeros((*m.shape,3),np.uint8);color[:]=(60,100,140);copy=color.copy()
        r=OuterContourDetector().detect(m,diagnostics=False);a=r.render_debug(color,panel=False)
        b=r.render_debug(color[...,::-1],color_order='BGR',panel=False)
        np.testing.assert_array_equal(a,b[...,::-1]);np.testing.assert_array_equal(color,copy)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'debug.png';self.assertEqual(r.save_debug(path,color,panel=False),path)
            np.testing.assert_array_equal(cv2.imread(str(path))[...,::-1],a)
            with self.assertRaises(FileExistsError):r.save_debug(path,color)
        empty=OuterContourDetector().detect(np.zeros_like(m));self.assertEqual(empty.render_debug(panel=False).shape,(*m.shape,3))
        with self.assertRaises(ValueError):r.render_debug(np.zeros((2,2,3),np.uint8))

    def test_noise_and_resize_coverage(self):
        m=rectangle();rng=np.random.default_rng(42)
        for x in range(36,155):m[30:32,x]=rng.random(2)>.5
        check_partition(self,OuterContourDetector().detect(m))
        large=cv2.resize(rectangle().astype(np.uint8),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)>0
        r=OuterContourDetector(smoothing_px=2,corner_window_px=16,min_corner_spacing_px=16,line_error_px=3,min_section_length_px=16,min_straight_length_px=40).detect(large)
        check_partition(self,r);self.assertEqual(len(r.contours[0]['corners']),4)


if __name__=='__main__':unittest.main()
