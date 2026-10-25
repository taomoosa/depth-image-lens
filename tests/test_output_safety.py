import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from depth_image_lens.diagnostics import save_debug, _atomic_write
from .test_bridge_drape import scene, inputs, ROIS


class OutputSafetyTests(unittest.TestCase):
    def result(self, kind='u'):
        return BridgeDrapeDetector().detect(*inputs(scene(kind)),ROIS,diagnostics=True)

    def test_each_output_symlink_is_replaced_without_touching_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'out';out.mkdir()
            for name in ('result.json','arrays.npz','overlay.svg'):
                with self.subTest(name=name):
                    marker=root/('outside-'+name);marker.write_bytes(b'untouched marker')
                    destination=out/name
                    destination.unlink(missing_ok=True)
                    destination.symlink_to(marker)
                    save_debug(self.result(),out)
                    self.assertEqual(marker.read_bytes(),b'untouched marker')
                    self.assertFalse(destination.is_symlink());self.assertTrue(destination.is_file())
            self.assertFalse(list(out.glob('.depth-debug-*')))

    def test_output_directory_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/'target';target.mkdir()
            marker=target/'marker';marker.write_text('unchanged')
            link=root/'out';link.symlink_to(target,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'directory must not be a symlink'):
                save_debug(self.result(),link)
            self.assertEqual(marker.read_text(),'unchanged')
            self.assertEqual({p.name for p in target.iterdir()},{'marker'})

    def test_regular_outputs_are_reusable_and_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'out'
            save_debug(self.result(),out)
            self.assertEqual(json.loads((out/'result.json').read_text())['bridge'],'yes')
            save_debug(self.result('left'),out)
            self.assertEqual(json.loads((out/'result.json').read_text())['drape'],'left')
            with np.load(out/'arrays.npz',allow_pickle=False) as arrays:
                self.assertFalse(arrays['bridge_corridor_mask'].any())
            ET.parse(out/'overlay.svg')
            self.assertEqual({p.name for p in out.iterdir()},{'result.json','arrays.npz','overlay.svg'})

    def test_no_overwrite_atomic_publication_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);marker=root/'marker';marker.write_bytes(b'unchanged')
            link=root/'result.json';link.symlink_to(marker)
            with self.assertRaises(FileExistsError):_atomic_write(link,lambda stream:stream.write(b'new'),overwrite=False)
            self.assertEqual(marker.read_bytes(),b'unchanged');self.assertTrue(link.is_symlink())
            self.assertFalse(list(root.glob('.depth-debug-*')))


if __name__=='__main__':unittest.main()
