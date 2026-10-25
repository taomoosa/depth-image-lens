import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from tests.test_bridge_drape import scene, inputs, ROIS


class SuppliedMaskTests(unittest.TestCase):
    def test_complete_mask_matches_geometry_without_difference(self):
        d=BridgeDrapeDetector()
        for kind in ('u','flat','left','separate'):
            mask=scene(kind);z,delta=inputs(mask)
            old=d.detect(z,delta,ROIS,diagnostics=True)
            new=d.detect_mask(z,mask,ROIS,depth_unit='m',mask_complete=True,diagnostics=True)
            self.assertEqual((old.bridge,old.large_sag,old.drape),(new.bridge,new.large_sag,new.drape))
            np.testing.assert_array_equal(old.foreground_mask,new.foreground_mask)
            self.assertFalse(new.features['measured_background_used'])
            self.assertEqual(new.features['difference_mode'],'supplied_mask')

    def test_unreviewed_mask_cannot_prove_absence_or_complete_upper_envelope(self):
        d=BridgeDrapeDetector();mask=scene('left');z,_=inputs(mask)
        r=d.detect_mask(z,mask,ROIS,depth_unit='m',diagnostics=True)
        self.assertEqual((r.bridge,r.drape),('unknown','unknown'))
        mask=scene('u');z,_=inputs(mask);r=d.detect_mask(z,mask,ROIS,depth_unit='m',diagnostics=True)
        self.assertEqual((r.bridge,r.large_sag,r.drape),('yes','yes','both'))
        self.assertTrue(r.diagnostics['sag_path_used'])
        self.assertIn('mask_completeness_unverified',r.stage_results['sag']['warnings'])

    def test_missing_anchors_return_unknown_without_invented_second_hand(self):
        mask=scene('left');z,_=inputs(mask)
        r=BridgeDrapeDetector().detect_mask(z,mask,depth_unit='m',diagnostics=True)
        self.assertEqual((r.bridge,r.large_sag,r.drape),('unknown','unknown','unknown'))
        self.assertEqual(r.reasons['bridge'],'supplied_mask_requires_two_anchor_rois')
        np.testing.assert_array_equal(r.foreground_mask,mask)

    def test_invalid_depth_and_excluded_mask_are_distinct_and_inputs_unchanged(self):
        mask=scene('u');z,_=inputs(mask);z[55,60]=0;observed=np.ones_like(mask);observed[56,60]=False
        before=z.copy();r=BridgeDrapeDetector().detect_mask(z,mask,ROIS,depth_unit='m',observed_mask=observed,diagnostics=True)
        self.assertTrue(r.invalid_mask[55,60]);self.assertTrue(r.invalid_mask[56,60])
        self.assertTrue(r.diagnostics['sensor_invalid_mask'][55,60])
        self.assertFalse(r.diagnostics['sensor_invalid_mask'][56,60])
        self.assertTrue(r.diagnostics['caller_unobserved_mask'][56,60])
        self.assertFalse(r.diagnostics['caller_unobserved_mask'][55,60])
        self.assertTrue(r.diagnostics['supplied_mask_excluded_mask'][0,0])
        self.assertFalse(r.invalid_mask[0,0]);np.testing.assert_array_equal(z,before)

    def test_explicit_units_and_no_numeric_mask_or_gap_inference(self):
        m=scene();z,_=inputs(m);d=BridgeDrapeDetector()
        a=d.detect_mask(z,m,ROIS,depth_unit='m',mask_complete=True)
        b=d.detect_mask(z*1000,m,ROIS,depth_unit='mm',mask_complete=True)
        self.assertEqual(a.to_dict(),b.to_dict())
        for kwargs in ({'depth_unit':'counts'},{'depth_unit':'m','mask_complete':1}):
            with self.assertRaises(ValueError):d.detect_mask(z,m,ROIS,**kwargs)
        with self.assertRaises(ValueError):d.detect_mask(z,m.astype(int),ROIS,depth_unit='m')
        with self.assertRaises(ValueError):BridgeDrapeDetector(max_gap_px=1).detect_mask(z,m,ROIS,depth_unit='m')


if __name__=='__main__':unittest.main()
