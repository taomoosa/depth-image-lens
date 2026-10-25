"""Feature independence regressions; synthetic geometry is not field accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector, SeededColorAssistant, analyze_color_geometry, LiftDetector, CameraIntrinsics, TablePlane, OverheadCornerDetector
from tests.test_bridge_drape import inputs, scene, ROIS


class IndependentFeaturesTests(unittest.TestCase):
    def test_auto_outlier_preserves_alternative_observed_path(self):
        m=scene('long_hem');z,d=inputs(m);z[45,60]=1.2
        detector=BridgeDrapeDetector()
        auto=detector.detect_auto(z,d,upper_band=(0,19),diagnostics=True)
        manual=detector.detect(z,d,ROIS,diagnostics=True)
        self.assertEqual((auto.bridge,auto.large_sag,auto.drape),('yes','no','both'))
        self.assertEqual(auto.bridge,manual.bridge)
        self.assertEqual(auto.stage_results['sag']['status'],'computed')
        self.assertFalse(auto.diagnostics['feature_observed_bridge_mask'][45,60])
        self.assertEqual(auto.stage_results['depth_connectivity']['rejected_edge_count'],8)
        evaluation=auto.auto['candidate_evaluations'][0]
        self.assertFalse(evaluation['depth_continuous'])  # Legacy whole-component check.
        self.assertEqual(evaluation['observed_route'],'yes')
        self.assertEqual(evaluation['decision'],'selected')
        self.assertEqual(auto.stage_results['grasp']['value'],'unknown')

    def test_auto_complete_depth_cut_keeps_bands_and_separate_components(self):
        m=scene('long_hem');z,d=inputs(m);z[:,60:][m[:,60:]]=1.2
        r=BridgeDrapeDetector().detect_auto(z,d,upper_band=(0,19),diagnostics=True)
        self.assertEqual(r.bridge,'unknown')
        self.assertEqual(r.auto['stage'],'depth_discontinuous')
        self.assertEqual(len(r.auto['hand_roi_candidates']),2)
        self.assertEqual(r.stage_results['observed_bridge']['status'],'computed')
        self.assertFalse(r.diagnostics['feature_observed_bridge_mask'].any())
        self.assertEqual(len(r.stage_results['depth_connectivity']['component_sizes']),2)
        self.assertFalse(any(c['spans_candidate_bands'] for c in r.stage_results['observed_bridge']['candidates']))
        self.assertEqual(r.stage_results['attachment_drape']['status'],'computed')
        self.assertEqual(r.stage_results['morphology']['status'],'computed')
        self.assertEqual(r.auto['candidate_evaluations'][0]['decision'],'rejected_depth_discontinuity')

    def test_whole_lift_requires_minimum_finite_observation_count(self):
        z=np.ones((10,10));k=CameraIntrinsics(100,100,5,5);plane=TablePlane((0,0,-1),1.2)
        for count in (1,12,19,20):
            m=np.zeros(z.shape,bool);m.flat[:count]=True
            r=LiftDetector().detect(z,k,plane,m,whole_object_visible=True)
            self.assertAlmostEqual(r.height_quantiles_m[1],.2)
            self.assertEqual(r.stage_results['height']['status'],'computed')
            self.assertEqual(r.whole_object_lifted,'yes' if count>=20 else 'unknown')
            if count<20:
                self.assertEqual(r.state,'unknown')
                self.assertIn('limited_observed_object_support',r.stage_results['whole_object']['warnings'])
            self.assertEqual(r.stage_results['whole_object']['finite_points'],count)

    def test_whole_lift_requires_positive_component_support(self):
        z=np.ones((10,10));m=np.zeros(z.shape,bool);m[::2,::2]=True
        r=LiftDetector().detect(z,CameraIntrinsics(100,100,5,5),TablePlane((0,0,-1),1.2),m,whole_object_visible=True)
        self.assertEqual(r.state,'unknown');self.assertEqual(r.whole_object_lifted,'unknown')
        self.assertEqual(r.stage_results['height']['points'],25)
        self.assertEqual(r.largest_elevated_component_pixels,1)
        self.assertIn('insufficient_elevated_component',r.stage_results['whole_object']['warnings'])

    def test_lift_finite_support_is_rechecked_after_deprojection(self):
        z=np.ones((10,3));k=CameraIntrinsics(1e-308,100,0,0);plane=TablePlane((0,0,-1),1.2)
        m=np.ones(z.shape,bool)
        r=LiftDetector(min_pixels=21).detect(z,k,plane,m,whole_object_visible=True)
        self.assertEqual(r.stage_results['height']['points'],20)
        self.assertAlmostEqual(r.height_quantiles_m[1],.2)
        self.assertEqual((r.state,r.whole_object_lifted),('unknown','unknown'))
        self.assertIn('some_nonfinite_deprojection',r.stage_results['whole_object']['warnings'])
        self.assertIn('limited_observed_object_support',r.stage_results['whole_object']['warnings'])
        m[:,2]=False
        r=LiftDetector().detect(z,k,plane,m,whole_object_visible=True)
        self.assertEqual((r.state,r.whole_object_lifted),('lifted','yes'))
        json.dumps(r.to_dict(),allow_nan=False)

    def test_lift_no_valid_depth_retains_all_stage_statuses(self):
        z=np.full((10,10),np.nan)
        r=LiftDetector().detect(z,CameraIntrinsics(100,100,5,5),TablePlane((0,0,-1),1.2),np.ones(z.shape,bool),whole_object_visible=True)
        self.assertEqual(set(r.stage_results),{'height','visible_lift','whole_object'})
        self.assertTrue(all(v['status']=='insufficient_input' for v in r.stage_results.values()))
        self.assertEqual(r.whole_object_lifted,'unknown')
        json.dumps(r.to_dict(),allow_nan=False)

    def test_partial_and_truncated_roi_preserve_observed_path_and_sag(self):
        m=scene('u');z,_=inputs(m)
        boxes={k:(x0,-12,x1,y1) for k,(x0,y0,x1,y1) in ROIS.items()}
        r=BridgeDrapeDetector().detect_mask(z,m,boxes,depth_unit='m',hand_visibility=('partial','partial'),diagnostics=True)
        self.assertEqual(r.bridge,'yes');self.assertEqual(r.large_sag,'yes')
        self.assertEqual(r.stage_results['observed_bridge']['status'],'computed')
        self.assertIn('left_roi_truncated',r.stage_results['observed_bridge']['warnings'])
        self.assertEqual(r.stage_results['grasp']['value'],'unknown')
        self.assertEqual(r.features['conservative_decisions']['bridge'],'unknown')
        self.assertFalse(r.diagnostics['inferred_gap_mask'].any())
        json.dumps(r.to_dict(),allow_nan=False)

    def test_local_bad_depth_edge_keeps_alternative_observed_route(self):
        m=scene('long_hem');z,_=inputs(m);z[45,60]=2.8
        r=BridgeDrapeDetector().detect_mask(z,m,ROIS,depth_unit='m',diagnostics=True)
        self.assertEqual(r.bridge,'yes')
        self.assertGreater(r.stage_results['depth_connectivity']['rejected_edge_count'],0)
        self.assertFalse(r.diagnostics['feature_observed_bridge_mask'][45,60])
        self.assertTrue(r.diagnostics['feature_depth_rejected_edge_mask'][45,60])
        self.assertEqual(r.stage_results['hanging']['value'],'present')

    def test_full_missing_cut_does_not_block_other_measurements(self):
        m=scene('u');z,_=inputs(m);z[:,57:64]=0
        r=BridgeDrapeDetector().detect_mask(z,m,ROIS,depth_unit='m')
        self.assertEqual(r.bridge,'unknown')
        self.assertEqual(r.stage_results['morphology']['status'],'computed')
        self.assertEqual(r.stage_results['sag']['status'],'inferred')
        self.assertIsNotNone(r.stage_results['sag']['ratio'])
        self.assertIn('some_object_depth_unobserved',r.stage_results['sag']['warnings'])

    def test_total_missing_depth_keeps_supplied_silhouette_not_bridge(self):
        m=scene('u');r=BridgeDrapeDetector().detect_mask(np.zeros(m.shape),m,depth_unit='m')
        self.assertEqual(r.stage_results['depth_connectivity']['status'],'insufficient_input')
        self.assertEqual(r.stage_results['morphology']['status'],'computed')
        self.assertEqual(r.stage_results['hanging']['status'],'inferred')
        self.assertEqual(r.bridge,'unknown')
        json.dumps(r.to_dict(),allow_nan=False)

    def test_one_neck_two_necks_and_identical_ambiguous_crop(self):
        yy,xx=np.indices((80,100));one=(np.abs(xx-50)<=2+yy*.4)&(yy<70)
        two=np.zeros_like(one);two[:50,15:25]=True;two[:50,75:85]=True;two[45:65,15:85]=True
        broad=np.zeros_like(one);broad[:70,20:80]=True
        for mask,hint in ((one,1),(two,2),(broad,None)):
            z=np.where(mask,1.,2.)
            r=BridgeDrapeDetector().detect_mask(z,mask,depth_unit='m')
            support=r.stage_results['support_hypotheses']
            self.assertEqual(support['preferred_count_hint'],hint)
            self.assertEqual(support['possible_counts'],[1,2]);self.assertFalse(support['unique_support_count'])
            self.assertEqual(support['hand_identity'],'unresolved')
            self.assertEqual(r.stage_results['sag']['value'],'unknown')
            self.assertIn('ratio',r.stage_results['sag'])
            self.assertTrue(all(ray['direction_xy'][1]<0 for ray in support['rays']))
            self.assertEqual(r.bridge,'unknown')
        # Exactly identical visible masks cannot reveal which hidden support layout produced them.
        a=BridgeDrapeDetector().detect_mask(np.where(broad,1.,2.),broad,depth_unit='m')
        b=BridgeDrapeDetector().detect_mask(np.where(broad,1.,2.),broad.copy(),depth_unit='m')
        self.assertEqual(a.to_dict(),b.to_dict())

    def test_local_depth_failure_keeps_color_drape_checks_on_new_components(self):
        from tests.rgbd_route_cases import route_cases
        from examples.example_rgbd import run
        c=next(c for c in route_cases() if c['name']=='curved_unknown_branch')
        c['depth'][70,88]=2600
        _,colors,r=run(c['depth'],c['difference'],c['color'],c['hand_rois'],
            c['object_seeds'],c['hand_seeds'],color_order='RGB',registered_to_depth=True,
            semantic_path_policy='observed_route')
        self.assertEqual(r.bridge,'yes')
        self.assertEqual(r.drape,'unknown')
        self.assertTrue((r.diagnostics['left_drape_component_mask']&colors.appearance_uncertain_mask).any())
        self.assertEqual(r.stage_results['attachment_drape']['status'],'computed')
        self.assertEqual(r.stage_results['semantic_classification']['drape'],'unknown')

    def test_static_assumption_can_be_disabled(self):
        m=scene('left');z,_=inputs(m)
        r=BridgeDrapeDetector(static_hanging=False).detect_mask(z,m,depth_unit='m')
        self.assertEqual(r.stage_results['support_hypotheses']['status'],'insufficient_input')
        self.assertEqual(r.stage_results['morphology']['status'],'computed')

    def test_robot_exclusion_never_becomes_object_bridge(self):
        m=scene('flat')|scene('left');z,_=inputs(m);arm=scene('flat')
        r=BridgeDrapeDetector().detect_mask(z,m,ROIS,depth_unit='m',arm_mask=arm)
        self.assertNotEqual(r.bridge,'yes');self.assertEqual(r.stage_results['hanging']['status'],'inferred')

    def test_incomplete_mask_preserves_measurement_with_warning(self):
        m=scene('u');z,_=inputs(m)
        r=BridgeDrapeDetector().detect_mask(z,m,ROIS,depth_unit='m',mask_complete=False)
        self.assertEqual(r.large_sag,'yes')
        self.assertIn('mask_completeness_unverified',r.stage_results['sag']['warnings'])
        self.assertEqual(r.stage_results['grasp']['value'],'unknown')

    def test_single_color_seed_class_keeps_appearance_hypothesis(self):
        m=scene('u');z,_=inputs(m);rgb=np.zeros((*m.shape,3),np.uint8);rgb[m]=(30,80,190)
        seeds=np.zeros_like(m);seeds[55,60]=True
        colors=SeededColorAssistant().segment(rgb,m,object_seeds=seeds,hand_seeds=np.zeros_like(m),color_order='RGB',registered_to_depth=True)
        self.assertFalse(colors.object_mask.any())
        self.assertTrue(colors.object_appearance_candidate_mask.any())
        result=analyze_color_geometry(BridgeDrapeDetector(),z,colors,ROIS,depth_unit='m')
        self.assertEqual(result.stage_results['appearance']['status'],'computed')
        self.assertFalse(result.stage_results['appearance']['class_separation_available'])
        self.assertEqual(result.stage_results['morphology']['status'],'computed')
        self.assertEqual(result.stage_results['grasp']['value'],'unknown')

    def test_auto_upper_crop_retains_support_and_morphology_stages(self):
        m=scene('separate')[33:];z,d=inputs(m)
        r=BridgeDrapeDetector().detect_auto(z,d,upper_band=(0,5))
        self.assertEqual(r.bridge,'unknown')
        self.assertEqual(r.stage_results['support_hypotheses']['preferred_count_hint'],2)
        self.assertEqual(r.stage_results['morphology']['status'],'computed')

    def test_lift_low_coverage_keeps_height_statistics(self):
        m=np.ones((20,20),bool);z=np.ones((20,20));z[:18]=0
        r=LiftDetector(min_valid_fraction=.7).detect(z,CameraIntrinsics(100,100,10,10),TablePlane((0,0,-1),1.2),m)
        self.assertEqual(r.state,'unknown');self.assertAlmostEqual(r.height_quantiles_m[1],.2)
        self.assertEqual(r.stage_results['height']['status'],'computed')
        self.assertIn('limited_observed_object_support',r.stage_results['height']['warnings'])

    def test_corner_nonright_angle_still_computes_depth(self):
        y,x=np.indices((100,100));m=(x>=20)&(y>=20)&(x+y<=130)
        result=OverheadCornerDetector().detect(np.zeros((100,100,3),np.uint8),np.ones((100,100)),color_order='RGB',registered_to_depth=True,object_mask=m,depth_scale_m=1.)
        rejected=[c for c in result.candidates if 'not_convex_near_right_angle' in c['reasons']]
        self.assertTrue(rejected)
        self.assertTrue(any(c['stage_results']['depth']['status']=='computed' for c in rejected))

if __name__=='__main__':unittest.main()
