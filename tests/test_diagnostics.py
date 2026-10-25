import json
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from .test_bridge_drape import scene, inputs, ROIS
from .phase1_cases import cases


class DiagnosticsTests(unittest.TestCase):
    def test_frozen_derived_scenes(self):
        for case in cases():
            with self.subTest(case=case['name']):
                result=BridgeDrapeDetector().detect_auto(case['depth'],case['difference'],**case['options'])
                expected='unknown' if case['name']=='separate_missing' else case['bridge_GT']
                self.assertEqual(result.bridge,expected)
                if expected=='yes':self.assertEqual(result.large_sag,'yes' if case['name'].startswith('u_') else 'no')

    def test_isolated_upper_speck_does_not_veto_bridge(self):
        mask=scene('u');mask[0,0]=True
        result=BridgeDrapeDetector().detect_auto(*inputs(mask),upper_band=(8,20))
        self.assertEqual((result.bridge,result.large_sag),('yes','yes'))
        mask[:30,0]=True
        self.assertEqual(BridgeDrapeDetector().detect_auto(*inputs(mask),upper_band=(8,20)).bridge,'unknown')

    def test_sloped_full_pixel_coverage(self):
        mask=np.zeros((100,120),bool)
        for x in range(20,101):
            y=20+round((x-20)*.1);mask[y:y+7,x]=True
        result=BridgeDrapeDetector().detect_auto(*inputs(mask),upper_band=(8,20))
        self.assertEqual((result.bridge,result.large_sag),('yes','no'))
        self.assertEqual(result.features['span_coverage'],1.)

    def test_diagnostics_are_opt_in_and_do_not_change_decisions(self):
        detector=BridgeDrapeDetector();z,diff=inputs(scene('u'))
        for method,kwargs in [(detector.detect,{'hand_rois':ROIS}),(detector.detect_auto,{'upper_band':(8,20)})]:
            plain=method(z,diff,**kwargs);debug=method(z,diff,diagnostics=True,**kwargs)
            value=debug.to_dict();self.assertIn('diagnostics',value);value.pop('diagnostics')
            self.assertEqual(value,plain.to_dict())
            json.dumps(debug.to_dict(include_masks=True),allow_nan=False)
            json.dumps(debug.to_dict(),allow_nan=False)
            diag=debug.diagnostics
            self.assertTrue(diag['sag_path_used']);self.assertTrue(diag['bridge_corridor_mask'].any())
            self.assertFalse(np.any(diag['bridge_corridor_mask'] & ~debug.foreground_mask))
            self.assertTrue(np.isfinite(diag['sag_path_xy']).all())

    def test_unknown_opposite_has_search_mask_not_fabricated_anchor(self):
        result=BridgeDrapeDetector().detect_auto(*inputs(scene('left')),upper_band=(8,20),diagnostics=True)
        self.assertEqual(result.drape,'unknown')
        self.assertEqual(set(result.auto['hand_roi_candidates']),{'observed'})
        self.assertTrue(result.diagnostics['opposite_search_mask'].any())
        self.assertNotIn('sag_path_xy',result.diagnostics)

    def test_missing_depth_is_not_invented_in_corridor(self):
        z,diff=inputs(scene());z[:,57:64]=0
        result=BridgeDrapeDetector().detect(z,diff,ROIS,diagnostics=True)
        self.assertEqual(result.bridge,'unknown');self.assertFalse(result.diagnostics['bridge_corridor_mask'].any())
        self.assertFalse(result.diagnostics['sag_path_used'])
        with self.assertRaises(ValueError):BridgeDrapeDetector().detect_auto(z,diff,diagnostics='true')


if __name__=='__main__':unittest.main()
