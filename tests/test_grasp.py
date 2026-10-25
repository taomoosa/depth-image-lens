"""Independent hand/object proximity tests, not physical grasp accuracy."""
import json
import unittest
import numpy as np
from depth_image_lens import IndependentGraspDetector,SeededColorAssistant


def scene(left=True,right=True):
    shape=(60,100);obj=np.zeros(shape,bool);obj[25:50,30:70]=True
    a=np.zeros(shape,bool);a[20:35,20 if left else 10:30 if left else 20]=True
    b=np.zeros(shape,bool);b[20:35,70 if right else 80:80 if right else 90]=True
    z=np.full(shape,2.);z[obj]=1.;z[a|b]=1.01
    return z,obj,{'left':a,'right':b}


def detect(z,obj,hands,**options):
    return IndependentGraspDetector().detect(z,obj,hands,depth_unit='m',registered_to_depth=True,
        mask_complete=(True,True),diagnostics=True,**options)


class GraspTests(unittest.TestCase):
    @staticmethod
    def contact_regions(a=True,b=True,valid_region=False):
        z=np.ones((30,100));obj=np.zeros(z.shape,bool);left=np.zeros_like(obj);right=np.zeros_like(obj)
        if a:left[10:13,10]=True;obj[11,12]=True
        if b:
            left[10,50]=left[12,50]=left[11,49]=True;obj[10:13,52]=True
        if valid_region:left[20:24,30]=True;obj[20:24,32]=True
        right[15:19,80]=True;obj[15:19,82]=True
        return z,obj,{'left':left,'right':right}

    def test_unrelated_contact_regions_cannot_mix_support_counts(self):
        opposite=None
        for a,b in ((True,False),(False,True),(True,True)):
            r=detect(*self.contact_regions(a,b));side=r.sides['left']
            self.assertEqual(side['estimate'],'unknown')
            self.assertFalse(side['stages']['grasp_inference']['checks']['corresponding_region_sufficient'])
            regions=side['stages']['depth_consistency']['corresponding_contact_regions']
            self.assertTrue(all(region['hand_pixels']<3 or region['object_pixels']<3 for region in regions))
            self.assertEqual(r.sides['right']['estimate'],'likely')
            if opposite is None:opposite=r.sides['right']
            else:self.assertEqual(r.sides['right'],opposite)
            if a and b:
                d=side['stages']['depth_consistency']
                self.assertEqual((d['hand_contact_component_pixels'],d['object_contact_component_pixels']),(3,3))
                self.assertIn('support_split_across_contact_regions',side['warnings'])
            json.dumps(r.to_dict(),allow_nan=False)

    def test_valid_corresponding_region_survives_other_weak_regions(self):
        r=detect(*self.contact_regions(valid_region=True));side=r.sides['left']
        self.assertEqual(side['estimate'],'likely');self.assertEqual(r.sides['right']['estimate'],'likely')
        regions=side['stages']['depth_consistency']['corresponding_contact_regions']
        self.assertTrue(any(region['hand_pixels']>=3 and region['object_pixels']>=3 for region in regions))
        self.assertTrue(any(region['hand_pixels']<3 or region['object_pixels']<3 for region in regions))
        self.assertTrue(side['stages']['grasp_inference']['checks']['corresponding_region_sufficient'])

    def test_both_left_right_neither_independent_of_bridge(self):
        for a,b in ((True,True),(True,False),(False,True),(False,False)):
            z,obj,hands=scene(a,b);r=detect(z,obj,hands)
            self.assertEqual([r.sides[k]['estimate'] for k in ('left','right')],
                             ['likely' if x else 'unlikely' for x in (a,b)])
            self.assertTrue(all(s['physical_grasp']=='unknown' for s in r.sides.values()))
            json.dumps(r.to_dict(),allow_nan=False)
        z,obj,hands=scene();obj[:,45:55]=False  # No spanning object region is required.
        self.assertEqual([s['estimate'] for s in detect(z,obj,hands).sides.values()],['likely','likely'])

    def test_near_but_depth_separated_is_not_positive(self):
        z,obj,hands=scene();z[hands['left']]=.8;r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'unlikely')
        self.assertEqual(r.sides['left']['reason'],'near_boundaries_depth_separated')
        self.assertEqual(r.sides['right']['estimate'],'likely')
        self.assertAlmostEqual(r.sides['left']['stages']['depth_consistency']['depth_delta_quantiles_m'][1],.2)

    def test_partial_hand_local_support_and_offscreen_hand(self):
        z,obj,hands=scene();r=detect(z,obj,hands,hand_visibility=('partial','visible'))
        self.assertEqual(r.sides['left']['estimate'],'likely')
        z[hands['left']]=.8;r=detect(z,obj,hands,hand_visibility=('partial','visible'))
        self.assertEqual(r.sides['left']['estimate'],'unknown')
        self.assertEqual(r.sides['left']['stages']['depth_consistency']['status'],'computed')
        hands['left'][:]=False;r=detect(z,obj,hands,hand_visibility=('hidden','visible'))
        self.assertEqual(r.sides['left']['estimate'],'unknown');self.assertEqual(r.sides['right']['estimate'],'likely')

    def test_holes_do_not_veto_opposite_hand_or_local_positive_support(self):
        z,obj,hands=scene();z[hands['left']]=0;r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'unknown');self.assertEqual(r.sides['right']['estimate'],'likely')
        self.assertEqual(r.sides['left']['stages']['boundary_proximity']['status'],'computed')
        z[25:29,29]=1.01;r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'likely')
        self.assertIn('contact_depth_coverage_low',r.sides['left']['warnings'])
        self.assertFalse((r.diagnostics['left']['consistent_hand_contact_mask']&(z<=0)).any())
        z[:]=np.nan;r=detect(z,obj,hands)
        self.assertTrue(all(s['estimate']=='unknown' for s in r.sides.values()))
        json.dumps(r.to_dict(),allow_nan=False)

    def test_overlap_and_local_unknown_labels_are_not_negative_evidence(self):
        z,obj,hands=scene();hands['left'][:,30:32]|=obj[:,30:32]
        r=detect(z,obj,hands);self.assertIn('ambiguous_hand_pixels_excluded',r.sides['left']['warnings'])
        self.assertEqual(r.sides['right']['estimate'],'likely')
        z,obj,hands=scene(False,True);u=np.zeros(z.shape,bool);u[:,20:23]=True
        r=detect(z,obj,hands,uncertain_mask=u)
        self.assertEqual(r.sides['left']['estimate'],'unknown');self.assertEqual(r.sides['right']['estimate'],'likely')

    def test_small_or_mixed_support_keeps_measurements_without_positive(self):
        z,obj,hands=scene();z[hands['left']]=.8;z[25,29]=1.01
        r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'unknown')
        self.assertGreater(r.sides['left']['stages']['depth_consistency']['consistent_hand_pixels'],0)
        self.assertEqual(r.sides['right']['estimate'],'likely')

    def test_nearest_depth_mismatch_not_bypassed_by_farther_match(self):
        z,obj,hands=scene();obj[:]=False;obj[25:35,30]=True;obj[25:35,32]=True
        z[obj]=1.3;z[25:35,32]=1.01
        r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'unlikely')
        self.assertEqual(r.sides['left']['stages']['depth_consistency']['consistent_hand_pixels'],0)

    def test_uncertain_completeness_preserves_negative_measurements(self):
        z,obj,hands=scene(False,False)
        r=IndependentGraspDetector().detect(z,obj,hands,depth_unit='m',registered_to_depth=True)
        self.assertTrue(all(s['estimate']=='unknown' for s in r.sides.values()))
        self.assertTrue(all(s['reason']=='no_object_boundary_within_radius' for s in r.sides.values()))

    def test_hidden_declaration_with_pixels_warns_without_suppressing_other_side(self):
        z,obj,hands=scene();r=detect(z,obj,hands,hand_visibility=('hidden','visible'))
        self.assertEqual(r.sides['left']['estimate'],'unknown');self.assertEqual(r.sides['right']['estimate'],'likely')
        self.assertGreater(r.sides['left']['stages']['depth_consistency']['consistent_hand_pixels'],0)

    def test_rgb_candidates_require_separated_hand_appearance(self):
        z,obj,hands=scene();fg=obj|hands['left']|hands['right'];rgb=np.zeros((*z.shape,3),np.uint8)
        rgb[obj]=(30,80,190);rgb[hands['left']|hands['right']]=(160,110,65)
        os=np.zeros(z.shape,bool);os[30,40]=True;hs=np.zeros_like(os);hs[25,25]=True
        colors=SeededColorAssistant().segment(rgb,fg,object_seeds=os,hand_seeds=hs,color_order='RGB',registered_to_depth=True)
        r=IndependentGraspDetector().detect_color_candidates(z,colors,hands,depth_unit='m')
        self.assertEqual([s['estimate'] for s in r.sides.values()],['likely','likely'])
        hs[:]=False;colors=SeededColorAssistant().segment(rgb,fg,object_seeds=os,hand_seeds=hs,color_order='RGB',registered_to_depth=True)
        r=IndependentGraspDetector().detect_color_candidates(z,colors,hands,depth_unit='m')
        self.assertTrue(all(s['estimate']=='unknown' for s in r.sides.values()))

    def test_units_strides_threshold_and_inputs_unchanged(self):
        z,obj,hands=scene();before=z.copy();d=IndependentGraspDetector()
        expected=detect(z,obj,hands).to_dict()
        raw=np.repeat((z*1000).astype(np.uint16),2,axis=1)[:,::2]
        r=d.detect_16uc1(raw,obj,hands,registered_to_depth=True,mask_complete=(True,True),diagnostics=True)
        self.assertEqual(r.to_dict(),expected)
        r=d.detect(raw,obj,hands,depth_unit='mm',registered_to_depth=True,mask_complete=(True,True))
        self.assertEqual(r.to_dict(),expected);np.testing.assert_array_equal(z,before)
        for delta,estimate in ((.04,'likely'),(.04+1e-10,'unlikely')):
            z[hands['left']]=1.+delta
            self.assertEqual(detect(z,obj,hands).sides['left']['estimate'],estimate)

    def test_invalid_contracts_and_empty_regions(self):
        z,obj,hands=scene();d=IndependentGraspDetector()
        for options in (dict(registered_to_depth=False),dict(depth_unit='counts'),dict(hand_visibility=True),dict(mask_complete=True)):
            args=dict(depth_unit='m',registered_to_depth=True);args.update(options)
            with self.assertRaises(ValueError):d.detect(z,obj,hands,**args)
        with self.assertRaises(ValueError):d.detect(z,obj.astype(int),hands,depth_unit='m',registered_to_depth=True)
        with self.assertRaises(ValueError):IndependentGraspDetector(contact_radius_px=20)
        obj[:]=False;r=detect(z,obj,hands)
        self.assertTrue(all(s['estimate']=='unknown' for s in r.sides.values()))
        json.dumps(r.to_dict(),allow_nan=False)


class LocalPatchTests(unittest.TestCase):
    def patches(self, bad_value=.5, contiguous=False):
        z=np.ones((40,90));obj=np.zeros(z.shape,bool);left=np.zeros_like(obj);right=np.zeros_like(obj)
        left[5:9,5]=True;obj[5:9,7]=True
        right[5:9,75]=True;obj[5:9,77]=True
        if contiguous:
            left[9:21,5]=True;obj[9:21,7]=True;z[9:21,5]=bad_value
        else:
            left[20:32,30]=True;obj[20:32,32]=True;z[20:32,30]=bad_value
        return z,obj,{'left':left,'right':right}

    def test_other_inconsistent_patch_does_not_dilute_positive(self):
        r=detect(*self.patches());side=r.sides['left']
        self.assertEqual(side['estimate'],'likely');self.assertEqual(r.sides['right']['estimate'],'likely')
        d=side['stages']['depth_consistency']
        self.assertEqual(d['consistent_fraction'],.25)
        self.assertEqual(sum(p['positive'] for p in d['contact_patches']),1)
        self.assertIn('local_positive_with_mixed_global_contact_evidence',side['warnings'])
        self.assertFalse(side['stages']['grasp_inference']['checks']['consistent_fraction_sufficient'])
        self.assertTrue(side['stages']['grasp_inference']['checks']['local_patch_sufficient'])
        json.dumps(r.to_dict(),allow_nan=False)

    def test_invalid_patch_preserves_other_patch_and_diagnostics(self):
        for value in (0,np.nan,np.inf):
            r=detect(*self.patches(value));d=r.sides['left']['stages']['depth_consistency']
            self.assertEqual(r.sides['left']['estimate'],'likely')
            self.assertEqual(d['valid_near_fraction'],.25)
            self.assertIn('no_valid_depth',[p['reason'] for p in d['contact_patches']])
            self.assertEqual(r.sides['right']['estimate'],'likely')

    def test_connected_inconsistency_is_not_split_to_force_positive(self):
        r=detect(*self.patches(contiguous=True));d=r.sides['left']['stages']['depth_consistency']
        self.assertEqual(r.sides['left']['estimate'],'unknown')
        self.assertEqual(len(d['contact_patches']),1)
        self.assertEqual(d['contact_patches'][0]['reason'],'mixed_depth_evidence')
        self.assertEqual(r.sides['right']['estimate'],'likely')

    def test_subthreshold_patches_do_not_pool(self):
        z,obj,hands=self.patches();hands['left'][:]=False;obj[:]=False
        for y,x in ((5,5),(20,30)):
            hands['left'][y:y+2,x]=True;obj[y:y+2,x+2]=True
        obj[5:9,77]=True;z[:]=1
        r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'unknown')
        self.assertEqual(r.sides['right']['estimate'],'likely')
        self.assertFalse(any(p['positive'] for p in r.sides['left']['stages']['depth_consistency']['contact_patches']))

    def test_depth_fragmented_support_inside_patch_does_not_pool(self):
        z,obj,hands=self.patches(contiguous=True);z[hands['left']]=.5
        z[5:7,5]=1;z[10:12,5]=1
        r=detect(z,obj,hands)
        self.assertEqual(r.sides['left']['estimate'],'unknown')
        self.assertEqual(r.sides['right']['estimate'],'likely')

    def test_all_contact_depth_missing_remains_unknown(self):
        z,obj,hands=self.patches();z[:]=0;r=detect(z,obj,hands)
        self.assertTrue(all(side['estimate']=='unknown' for side in r.sides.values()))
        self.assertTrue(all(not p['positive'] for side in r.sides.values()
            for p in side['stages']['depth_consistency']['contact_patches']))
