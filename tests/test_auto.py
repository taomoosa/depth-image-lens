import json
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from .auto_cases import cases
from .test_bridge_drape import scene, inputs


class AutoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows={};detector=BridgeDrapeDetector()
        for name,mask,_,options in cases():
            depth,difference=inputs(mask);kwargs={}
            if options.get('missing'):depth[:,57:64]=0
            if options.get('exclude'):kwargs['arm_mask']=mask
            if options.get('depth_step'):
                depth[mask&(np.indices(mask.shape)[1]>60)]=1.1;difference=2.-depth
            cls.rows[name]=detector.detect_auto(depth,difference,upper_band=(8,20),**kwargs).to_dict()
    def test_typical_bridges(self):
        for name in ('flat','deep_u','bridge_asymmetric_drape','sloped_height_within_band'):
            with self.subTest(name=name):self.assertEqual(self.rows[name]['bridge'],'yes')
        self.assertEqual(self.rows['deep_u']['large_sag'],'yes')
        self.assertEqual(self.rows['flat']['large_sag'],'no')
        f=self.rows['bridge_asymmetric_drape']['features']
        self.assertGreater(f['image_left_local_hanging_extent_px'],f['image_right_local_hanging_extent_px'])
    def test_unilateral_visible_opposite(self):
        self.assertEqual(self.rows['left_visible_opposite']['drape'],'left')
        self.assertEqual(self.rows['right_visible_opposite']['drape'],'right')
        self.assertEqual(self.rows['separate_drapes']['drape'],'both')
        self.assertEqual(self.rows['separate_drapes']['bridge'],'no')
    def test_no_invented_opposite(self):
        for name in ('left_opposite_absent','right_opposite_absent'):
            r=self.rows[name];self.assertEqual(r['drape'],'unknown')
            self.assertEqual(set(r['auto']['hand_roi_candidates']),{'observed'})
            self.assertIsNotNone(r['auto']['opposite_search_region'])
    def test_arm_upper_path_excluded(self):
        self.assertEqual(self.rows['arms_connected_above_band']['bridge'],'no')
        self.assertEqual(self.rows['arm_slab_with_exclusion']['bridge'],'unknown')
    def test_missing_height_crop_and_depth(self):
        for name in ('missing_depth_cut','height_assumption_violation','upper_cut','depth_step_bridge'):
            with self.subTest(name=name):self.assertEqual(self.rows[name]['bridge'],'unknown')
        self.assertNotEqual(self.rows['large_observed_gap']['bridge'],'yes')
    def test_contract_and_json(self):
        json.dumps(self.rows,allow_nan=False)
        z,diff=inputs(scene());d=BridgeDrapeDetector()
        with self.assertRaises(ValueError):d.detect_auto(z,diff,upper_band=(20,8))
        with self.assertRaises(ValueError):d.detect_auto(z,diff,upper_band=(200,300))
        with self.assertRaises(ValueError):d.detect_auto(z,diff,depth_unit='mm',difference_unit='m')
        with self.assertRaises(ValueError):d.detect_auto(z,diff,hand_rois={})
        self.assertEqual(d.detect_auto(z,diff).bridge,'yes')
    def test_manual_semantic_limit_not_hidden(self):
        # Same pixels/inputs as flat bridge: this known semantic error is reported.
        self.assertEqual(self.rows['arm_slab_without_semantic_exclusion']['bridge'],'yes')


if __name__=='__main__':unittest.main(verbosity=2)
