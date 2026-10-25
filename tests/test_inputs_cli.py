import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector, background_difference
from .test_bridge_drape import scene, inputs, ROIS


def strict_load(text):
    def reject(value):raise AssertionError('Nonfinite JSON constant: '+value)
    return json.loads(text, parse_constant=reject)


class InputTests(unittest.TestCase):
    def test_unsigned_subtraction_preserves_sign(self):
        current=np.array([[2500,1500],[2000,1000]],dtype=np.uint16)
        background=np.full((2,2),2000,dtype=np.uint16)
        difference,valid=background_difference(current,background,depth_unit='mm',background_unit='mm')
        self.assertEqual(difference.dtype,np.float64)
        self.assertEqual(difference.tolist(),[[-500.,500.],[0.,1000.]])
        self.assertTrue(valid.all())

    def test_m_and_mm_equivalent_detection(self):
        depth,_=inputs(scene('u'));background=np.full_like(depth,2.)
        a,va=background_difference(depth,background,depth_unit='m',background_unit='m')
        z=(depth*1000).astype(np.uint16);b,vb=background_difference(z,(background*1000).astype(np.uint16),depth_unit='mm',background_unit='mm')
        first=BridgeDrapeDetector().detect_auto(depth,a,upper_band=(8,20),background_valid=va)
        second=BridgeDrapeDetector().detect_auto(z,b,upper_band=(8,20),background_valid=vb,depth_unit='mm',difference_unit='mm')
        self.assertEqual(first.to_dict(),second.to_dict())

    def test_invalid_and_mismatched_inputs(self):
        current=np.array([[1.,0.],[65535.,np.nan]]);background=np.array([[0.,2.],[2.,2.]])
        difference,valid=background_difference(current,background,depth_unit='mm',background_unit='mm',invalid_values=[65535])
        self.assertFalse(valid.any());self.assertTrue(np.isnan(difference).all())
        for a,b,units in [(np.ones((2,2)),np.ones((3,2)),('m','m')),(np.ones((2,2)),np.ones((2,2)),('m','mm')),(np.ones((2,2),dtype=object),np.ones((2,2)),('m','m'))]:
            with self.assertRaises(ValueError):background_difference(a,b,depth_unit=units[0],background_unit=units[1])


class CLITests(unittest.TestCase):
    def invoke(self, directory, *arguments):
        command=Path(sys.executable).with_name('depth-image-lens')
        self.assertTrue(command.is_file(),'Install editable package before testing the console entrypoint')
        env=os.environ.copy();env.pop('PYTHONPATH',None);env['PYTHONDONTWRITEBYTECODE']='1'
        return subprocess.run([str(command),*arguments],cwd=directory,env=env,text=True,capture_output=True)

    def test_npz_background_auto_and_editable_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);depth,_=inputs(scene('u'))
            np.savez(root/'private_source_name.npz',depth=(depth*1000).astype(np.uint16),background=np.full(depth.shape,2000,dtype=np.uint16))
            (root/'config.json').write_text(json.dumps({'depth_unit':'mm','background_unit':'mm','upper_band':[8,20]}))
            p=self.invoke(tmp,'--input','private_source_name.npz','--config','config.json','--output','out')
            self.assertEqual(p.returncode,0,p.stderr);result=strict_load(p.stdout)
            self.assertEqual((result['bridge'],result['large_sag']),('yes','yes'))
            self.assertEqual(result,strict_load((root/'out/result.json').read_text()))
            svg=(root/'out/overlay.svg').read_text();self.assertNotIn('private_source_name',svg);self.assertNotIn(tmp,svg)
            self.assertNotIn(tmp,p.stdout)
            with np.load(root/'out/arrays.npz',allow_pickle=False) as data:self.assertTrue(data['bridge_corridor_mask'].any())

    def test_npy_difference_manual_and_npz_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);z,diff=inputs(scene('left'));np.save(root/'z.npy',z);np.save(root/'difference.npy',diff)
            cfg={'mode':'manual','hand_rois':ROIS,'depth_unit':'m','difference_unit':'m','difference_sign':'background-current'}
            (root/'config.json').write_text(json.dumps(cfg))
            p=self.invoke(tmp,'--depth','z.npy','--difference','difference.npy','--config','config.json','--output','out')
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(strict_load(p.stdout)['drape'],'left')
            np.savez(root/'pair.npz',depth=z,difference=diff)
            q=self.invoke(tmp,'--input','pair.npz','--config','config.json','--output','other')
            self.assertEqual(q.returncode,0,q.stderr);self.assertEqual(strict_load(p.stdout),strict_load(q.stdout))

    def test_cli_rejects_object_and_missing_units_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.savez(root/'object.npz',depth=np.array([['unsafe']]*2,dtype=object),background=np.ones((2,2)))
            np.savez(root/'numeric.npz',depth=np.ones((2,2)),background=np.ones((2,2)))
            (root/'config.json').write_text('{}')
            for name in ['object.npz','numeric.npz']:
                p=self.invoke(tmp,'--input',name,'--config','config.json','--output','out')
                self.assertNotEqual(p.returncode,0);self.assertFalse((root/'out').exists())
            (root/'config.json').write_text('{"depth_unit":"m","background_unit":"m","threshold":NaN}')
            p=self.invoke(tmp,'--input','numeric.npz','--config','config.json','--output','out')
            self.assertNotEqual(p.returncode,0);self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
