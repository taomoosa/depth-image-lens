import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from depth_image_lens.cli import run
from .test_bridge_drape import scene, ROIS


class PositiveForegroundTests(unittest.TestCase):
    def detect(self, difference, *, raw=None, auto=False, **kwargs):
        raw=np.full(difference.shape,1.5) if raw is None else raw
        detector=BridgeDrapeDetector(threshold_m=0)
        options={'difference_mode':'positive_foreground',**kwargs}
        if auto:return detector.detect_auto(raw,difference,upper_band=(8,20),**options)
        return detector.detect(raw,difference,ROIS,**options)

    def test_zero_is_valid_background_with_no_foreground(self):
        for auto in [False,True]:
            result=self.detect(np.zeros((100,120)),auto=auto)
            self.assertEqual((result.bridge,result.drape),('no','neither'))
            self.assertFalse(result.foreground_mask.any());self.assertFalse(result.invalid_mask.any())
            self.assertEqual(result.features['difference_mode'],'positive_foreground')

    def test_positive_bridge_left_drape_and_known_gap(self):
        for auto in [False,True]:
            bridge=self.detect(scene('u')*.5,auto=auto)
            self.assertEqual((bridge.bridge,bridge.large_sag),('yes','yes'))
        left=self.detect(scene('left')*.5)
        self.assertEqual((left.bridge,left.drape),('no','left'))
        mask=scene();mask[:,55:65]=False
        gap=self.detect(mask*.5)
        self.assertEqual(gap.bridge,'no');self.assertFalse(gap.invalid_mask.any())
        mask=scene();mask[:,60]=False
        small=BridgeDrapeDetector(threshold_m=0,max_gap_px=1).detect(np.full(mask.shape,1.5),mask*.5,ROIS,difference_mode='positive_foreground')
        self.assertEqual(small.bridge,'yes');self.assertTrue(small.features['gap_used'])

    def test_zero_threshold_excludes_zero_and_negative_without_background_reconstruction(self):
        difference=np.zeros((100,120));difference[40:50,40:50]=-100
        result=self.detect(difference)
        self.assertEqual(result.bridge,'no');self.assertFalse(result.foreground_mask.any())
        self.assertFalse(result.invalid_mask.any())
        # Signed mode still validates the implied physical background.
        signed=BridgeDrapeDetector(threshold_m=0).detect(np.full(difference.shape,1.5),difference,ROIS)
        self.assertTrue(signed.invalid_mask[45,45])

    def test_raw_missing_depth_is_not_known_zero_difference(self):
        difference=np.zeros((100,120))
        for value in [0.,float('nan')]:
            result=self.detect(difference,raw=np.full(difference.shape,value),auto=True)
            self.assertEqual(result.bridge,'unknown');self.assertTrue(result.invalid_mask.all())
        raw=np.full(difference.shape,1.5);raw[:,57:64]=0
        self.assertEqual(self.detect(scene()*.5,raw=raw).bridge,'unknown')

    def test_explicit_validity_false_keeps_unknown(self):
        for name in ['observed_mask','background_valid']:
            for auto in [False,True]:
                result=self.detect(np.zeros((100,120)),auto=auto,**{name:np.zeros((100,120),bool)})
                self.assertEqual(result.bridge,'unknown');self.assertTrue(result.invalid_mask.all())

    def test_explicit_units_and_mode_contract(self):
        d=scene('u')*.5;a=self.detect(d)
        b=self.detect(d*1000,raw=np.full(d.shape,1500),depth_unit='mm',difference_unit='mm')
        self.assertEqual(a.to_dict(),b.to_dict())
        with self.assertRaises(ValueError):self.detect(d,depth_unit='mm',difference_unit='m')
        with self.assertRaises(ValueError):self.detect(d,difference_sign='current-background')
        with self.assertRaises(ValueError):self.detect(d,difference_mode='other')

    def test_cli_masked_positive_without_sign_and_background_helper_separation(self):
        config={'difference_mode':'positive_foreground','depth_unit':'m','difference_unit':'m','upper_band':[8,20],'detector':{'threshold_m':0}}
        depth=np.full((100,120),1.5);difference=scene('u')*.5
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);np.savez(root/'pair.npz',depth=depth,difference=difference)
            (root/'config.json').write_text(json.dumps(config))
            command=Path(sys.executable).with_name('depth-image-lens')
            p=subprocess.run([str(command),'--input','pair.npz','--config','config.json','--output','out'],cwd=root,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr);result=json.loads(p.stdout)
            self.assertEqual((result['bridge'],result['large_sag']),('yes','yes'))
            self.assertEqual(result['features']['difference_mode'],'positive_foreground')
        with self.assertRaises(ValueError):run({'depth':depth,'background':depth},config,'.')


if __name__=='__main__':unittest.main()
