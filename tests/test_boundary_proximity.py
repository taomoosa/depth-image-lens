"""Observed-point proximity selection, not semantic boundary accuracy."""
import copy,json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from depth_image_lens import OverheadBoundaryDetector,BoundaryProximityResult


def segment(points,*,sid=1,evidence=None,status='candidate'):
    return dict(id=sid,component=1,points_xy=[list(p) for p in points],evidence=evidence or ['strong']*len(points),status=status,length_px=float(max(0,len(points)-1)),measurements={'darkness':{'mean':12.,'p95':15.}})


def result(paths,mask=None,graph_status='computed'):
    if mask is None:
        mask=np.zeros((100,120),bool);mask[10:90,10:110]=True
    base=OverheadBoundaryDetector().detect(np.full((*mask.shape,3),190,np.uint8),color_order='RGB',object_mask=mask)
    return replace(base,topology={'color':dict(status=graph_status,segments=paths,nodes=[],connections=[]),'depth':dict(status='computed',segments=[],nodes=[],connections=[])})


class BoundaryProximityTests(unittest.TestCase):
    def test_outer_line_and_interior(self):
        r=result([segment([(x,13) for x in range(15,105)]),segment([(x,50) for x in range(20,100)],sid=2)])
        f=r.filter_outer_proximity();self.assertIsInstance(f,BoundaryProximityResult)
        self.assertEqual([x['segment_id'] for x in f.excluded],[1]);self.assertEqual([x['segment_id'] for x in f.kept],[2]);self.assertEqual(f.excluded[0]['min_distance_px'],4.)

    def test_endpoint_only_contact_kept(self):
        f=result([segment([(x,50) for x in range(10,110)])]).filter_outer_proximity()
        self.assertEqual(len(f.kept),1);self.assertAlmostEqual(f.kept[0]['near_fraction'],.16)

    def test_threshold_directions_and_equality(self):
        r=result([segment([(x,13) for x in range(20,100)])])
        self.assertEqual(len(r.filter_outer_proximity(near_px=3.99).kept),1)
        self.assertEqual(len(r.filter_outer_proximity(near_px=4,reject_fraction=1.).excluded),1)
        r=result([segment([(x,50) for x in range(10,110)])])
        self.assertEqual(len(r.filter_outer_proximity(reject_fraction=.16).excluded),1)
        self.assertEqual(len(r.filter_outer_proximity(reject_fraction=.160001).kept),1)

    def test_hole_option(self):
        m=np.zeros((100,120),bool);m[10:90,10:110]=1;m[40:60,40:80]=0
        r=result([segment([(x,38) for x in range(42,78)])],m)
        self.assertEqual(len(r.filter_outer_proximity(include_holes=False).kept),1)
        self.assertEqual(len(r.filter_outer_proximity(include_holes=True).excluded),1)

    def test_inferred_points_not_support(self):
        points=[(x,50) for x in range(30,45)]+[(x,13) for x in range(30,100)]
        r=result([segment(points,evidence=['weak']*15+['inferred']*70)])
        f=r.filter_outer_proximity();self.assertEqual(len(f.kept),1);self.assertEqual(f.kept[0]['observed_count'],15);self.assertEqual(f.kept[0]['inferred_unique_count'],70)
        self.assertEqual(f.kept[0]['near_fraction'],0)

    def test_duplicate_observations_count_once(self):
        points=[(20,13)]*90+[(x,50) for x in range(30,40)]
        f=result([segment(points)]).filter_outer_proximity();self.assertEqual(f.kept[0]['observed_count'],11);self.assertAlmostEqual(f.kept[0]['near_fraction'],1/11)

    def test_all_inferred_and_short_paths(self):
        f=result([segment([(30,50),(31,50)],evidence=['inferred']*2),segment([(30,20)],sid=2,status='short_path')]).filter_outer_proximity()
        self.assertEqual(len(f.pending),1);self.assertEqual(len(f.ignored),1);self.assertIn('no_observed_points',f.pending[0]['reasons'])

    def test_invalid_points_pending_independently(self):
        paths=[segment([(-1,20),(20,20)]),segment([(20.5,30),(21,30)],sid=2),segment([(30,50),(31,50)],sid=3)]
        f=result(paths).filter_outer_proximity();self.assertEqual(len(f.pending),2);self.assertEqual([r['segment_id'] for r in f.kept],[3])

    def test_outside_object_and_domain_pending(self):
        r=result([segment([(2,50),(3,50)])]);f=r.filter_outer_proximity();self.assertEqual(len(f.pending),1)
        obs=np.ones(r.object_mask.shape,bool);obs[50,30]=0
        f=result([segment([(30,50),(31,50)])]).filter_outer_proximity(observed_mask=obs);self.assertEqual(len(f.pending),1)

    def test_frame_clipping_local_not_global_veto(self):
        m=np.zeros((100,120),bool);m[10:90,:110]=1
        r=result([segment([(2,y) for y in range(20,80)]),segment([(x,50) for x in range(30,70)],sid=2)],m)
        f=r.filter_outer_proximity();self.assertEqual([r['segment_id'] for r in f.pending],[1]);self.assertEqual([r['segment_id'] for r in f.kept],[2])

    def test_known_roi_cut_vs_undisclosed(self):
        m=np.zeros((100,120),bool);m[10:90,30:110]=1
        r=result([segment([(32,y) for y in range(20,80)])],m)
        self.assertEqual(len(r.filter_outer_proximity().excluded),1)
        obs=np.ones_like(m);obs[:,:30]=0
        f=r.filter_outer_proximity(observed_mask=obs);self.assertEqual(len(f.pending),1)

    def test_multicomponent_and_nested_island(self):
        m=np.zeros((120,140),bool);m[5:115,5:135]=1;m[20:100,20:120]=0;m[35:85,40:100]=1
        r=result([segment([(x,38) for x in range(45,95)])],m);f=r.filter_outer_proximity()
        self.assertEqual(len(f.excluded),1);self.assertEqual(f.excluded[0]['min_distance_px'],4.)
        f=result([segment([(10,50),(50,50)])],m).filter_outer_proximity();self.assertEqual(len(f.pending),1);self.assertIn('multiple_object_components',f.pending[0]['reasons'])

    def test_empty_masks_and_missing_topology(self):
        r=result([],np.zeros((100,120),bool));f=r.filter_outer_proximity();self.assertEqual(f.summary['status'],'no_segments')
        f=replace(r,topology={}).filter_outer_proximity();self.assertEqual(f.summary['channels']['color'],'not_available');self.assertEqual(f.records,[])
        self.assertEqual(len(result([segment([(20,20)])],np.zeros((100,120),bool)).filter_outer_proximity().pending),1)

    def test_no_mutation_and_detached_snapshot(self):
        r=result([segment([(x,50) for x in range(10,110)])]);before=copy.deepcopy(r.to_dict());mask=r.object_mask.copy()
        f=r.filter_outer_proximity();self.assertEqual(r.to_dict(),before);np.testing.assert_array_equal(mask,r.object_mask)
        f.records[0]['segment']['points_xy'][0][0]=99;self.assertEqual(r.to_dict(),before)
        json.dumps(f.to_dict(),allow_nan=False)

    def test_nonfinite_pending_json_and_debug(self):
        r=result([segment([(float('nan'),20),(30,30)])]);f=r.filter_outer_proximity()
        self.assertEqual(len(f.pending),1);self.assertEqual(f.pending[0]['invalid_input_point_count'],1)
        self.assertIsNone(f.to_dict()['records'][0]['segment']['points_xy'][0][0])
        json.dumps(f.to_dict(),allow_nan=False);f.render_debug()

    def test_invalid_configuration(self):
        r=result([])
        for kw in [dict(near_px=True),dict(reject_fraction=False),dict(near_px="8"),dict(near_px=-1),dict(near_px=np.inf),dict(near_px=[2]),dict(reject_fraction=0),dict(reject_fraction=1.1),dict(include_holes=1),dict(observed_mask=np.ones((2,2),bool))]:
            with self.assertRaises(ValueError):r.filter_outer_proximity(**kw)

    def test_debug_rgb_save_and_nonoverwrite(self):
        r=result([segment([(x,50) for x in range(10,110)])]);f=r.filter_outer_proximity();rgb=np.full((*r.object_mask.shape,3),(15,70,110),np.uint8);before=rgb.copy()
        a=f.render_debug(rgb,panel=False);b=f.render_debug(rgb[...,::-1],color_order='BGR',panel=False);np.testing.assert_array_equal(a,b[...,::-1]);np.testing.assert_array_equal(rgb,before)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.png';self.assertEqual(f.save_debug(p,rgb,panel=False),p);np.testing.assert_array_equal(cv2.imread(str(p))[...,::-1],a)
            with self.assertRaises(FileExistsError):f.save_debug(p,rgb)
        with self.assertRaises(ValueError):f.render_debug(np.zeros((2,2,3),np.uint8))

    def test_real_detector_channels_diagnostics_parity(self):
        m=np.zeros((100,140),bool);m[5:-5,5:-5]=1;color=np.full((*m.shape,3),190,np.uint8);color[12:15,20:120]=20;color[48:51,20:120]=20
        depth=np.full(m.shape,1500,np.uint16);depth[50:,20:120]=1530
        model=OverheadBoundaryDetector(trace_boundaries=True)
        a=model.detect(color,depth,color_order='RGB',registered_to_depth=True,object_mask=m,diagnostics=False)
        b=model.detect(color,depth,color_order='RGB',registered_to_depth=True,object_mask=m,diagnostics=True)
        self.assertEqual(a.filter_outer_proximity().to_dict(),b.filter_outer_proximity().to_dict())
        self.assertTrue(a.filter_outer_proximity().records);self.assertEqual(set(a.filter_outer_proximity().summary['channels']),{'color','depth'})

    def test_bad_evidence_and_uncomputed_topology(self):
        f=result([segment([(20,30)],evidence=['invalid'])]).filter_outer_proximity();self.assertEqual(len(f.pending),1)
        f=result([segment([(20,30)])],graph_status='insufficient').filter_outer_proximity();self.assertEqual(len(f.pending),1)


if __name__=='__main__':unittest.main()
