import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from depth_image_lens import (BridgeDrapeDetector, positive_difference_16uc1,
                              depth_pair_16uc1_to_meters)
from depth_image_lens.cli import run
from .test_bridge_drape import scene, ROIS


class Uint16ConvenienceTests(unittest.TestCase):
    def pair(self):
        mask=scene('u');depth=np.full(mask.shape,1500,dtype=np.uint16)
        return depth,(mask*500).astype(np.uint16)

    def test_clipped_subtraction_bounds_and_background_invalidity(self):
        z=np.array([[2500,1500,1],[65535,0,1000]],dtype=np.uint16)
        bg=np.array([[2000,2000,65535],[1,2000,0]],dtype=np.uint16)
        diff,valid=positive_difference_16uc1(z,bg)
        self.assertEqual(diff.dtype,np.uint16)
        self.assertEqual(diff.tolist(),[[0,500,65534],[0,0,0]])
        self.assertEqual(valid.tolist(),[[True,True,True],[True,False,False]])

    def test_default_and_explicit_scale_match_general_api(self):
        z,diff=self.pair();detector=BridgeDrapeDetector()
        for scale in [.001,.002]:
            a,b=depth_pair_16uc1_to_meters(z,diff,depth_scale_m=scale)
            self.assertEqual(a[0,0],1500*scale)
            for boxes in [None,ROIS]:
                options={'upper_band':(8,20)} if boxes is None else {}
                result=detector.detect_16uc1(z,diff,boxes,depth_scale_m=scale,**options)
                expected=(detector.detect_auto(a,b,difference_mode='positive_foreground',**options) if boxes is None else detector.detect(a,b,boxes,difference_mode='positive_foreground'))
                self.assertEqual(result.to_dict(),expected.to_dict())
        for bad in [0.,float('inf'),True]:
            with self.assertRaises(ValueError):depth_pair_16uc1_to_meters(z,diff,depth_scale_m=bad)
        with self.assertRaises(ValueError):detector.detect_16uc1(z.astype(float),diff)
        with self.assertRaises(ValueError):detector.detect_16uc1(z,diff,depth_unit='mm')

    def test_noncontiguous_views_are_read_only_inputs(self):
        z,diff=self.pair();base_z=np.repeat(z,2,axis=1);base_d=np.repeat(diff,2,axis=1)
        before_z=base_z.copy();before_d=base_d.copy();view_z=base_z[:,::2];view_d=base_d[:,::2]
        self.assertFalse(view_z.flags.c_contiguous)
        result=BridgeDrapeDetector().detect_16uc1(view_z,view_d,upper_band=(8,20))
        expected=BridgeDrapeDetector().detect_16uc1(z,diff,upper_band=(8,20))
        self.assertEqual(result.to_dict(),expected.to_dict())
        np.testing.assert_array_equal(base_z,before_z);np.testing.assert_array_equal(base_d,before_d)

    def test_zero_difference_vs_missing_background_mask(self):
        z=np.full((100,120),1500,dtype=np.uint16);diff=np.zeros_like(z)
        detector=BridgeDrapeDetector(threshold_m=0)
        result=detector.detect_16uc1(z,diff,upper_band=(8,20))
        self.assertEqual(result.bridge,'no');self.assertFalse(result.foreground_mask.any())
        prepared,valid=positive_difference_16uc1(z,np.zeros_like(z))
        self.assertFalse(valid.any());self.assertFalse(prepared.any())
        self.assertEqual(detector.detect_16uc1(z,prepared,background_valid=valid,upper_band=(8,20)).bridge,'unknown')
        self.assertEqual(detector.detect_16uc1(np.zeros_like(z),diff,upper_band=(8,20)).bridge,'unknown')

    def test_cli_encoding_for_masked_pair_and_empty_background(self):
        z,diff=self.pair();config={'encoding':'16UC1','upper_band':[8,20]}
        direct=run({'depth':z,'difference':diff},config,'.')
        self.assertEqual(direct.bridge,'yes')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);np.savez(root/'pair.npz',depth=z,background=z+diff)
            (root/'config.json').write_text(json.dumps(config))
            p=subprocess.run([str(Path(sys.executable).with_name('depth-image-lens')),'--input','pair.npz','--config','config.json','--output','out'],cwd=root,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),direct.to_dict())
        with self.assertRaises(ValueError):run({'depth':z,'difference':diff},{**config,'depth_unit':'mm'},'.')
        with self.assertRaises(ValueError):run({'depth':z,'difference':diff},{'depth_scale_m':.001},'.')


if __name__=='__main__':unittest.main()
