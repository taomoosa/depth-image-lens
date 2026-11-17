"""Mixed-resolution registered inputs share a normalized view, not a raw grid."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from depth_image_lens.tuning_backend import prepare_inputs, default_config, run_pipeline
from depth_image_lens.geometry import CameraIntrinsics
from tests.test_tuning_backend import scene


def large(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


class MixedResolutionTests(unittest.TestCase):
    def test_each_role_independently_and_cropped(self):
        a = scene()
        a.update(observed_mask=np.ones((400,640), bool), exclusion_mask=np.zeros((400,640), bool))
        expected = prepare_inputs(a, [40,40,80,90])
        for role in a:
            mixed = dict(a); mixed[role] = large(a[role])
            got = prepare_inputs(mixed, [40,40,80,90])
            for key in a:
                np.testing.assert_array_equal(got.arrays[key], expected.arrays[key], err_msg=role+' '+key)
            self.assertEqual(got.transform['sources'][role]['scale'], .5)

    def test_color_big_depth_small_values_intrinsics_and_coordinates(self):
        a = scene(); a['color'] = large(a['color']); a['background_color'] = large(a['background_color'])
        a['depth'] = a['depth'].astype(float)
        a['depth'][40,40:44] = [0, np.nan, np.inf, 987]
        p = prepare_inputs(a, [40,40,80,90], CameraIntrinsics(400,400,319.5,199.5))
        np.testing.assert_equal(p.arrays['depth'][0,:4], [0,np.nan,np.inf,987])
        self.assertEqual(p.intrinsics.fx, 400)
        self.assertEqual(p.intrinsics.cx, 279.5)
        self.assertEqual(p.transform['source_role'], 'depth')
        np.testing.assert_equal(p.to_source_xy([[0,0]]), [[40,40]])
        np.testing.assert_equal(p.to_source_xy([[0,0]], role='color'), [[80.5,80.5]])
        # Dictionary order cannot change the calibration reference.
        q = prepare_inputs(dict(reversed(list(a.items()))), [40,40,80,90], CameraIntrinsics(400,400,319.5,199.5))
        self.assertEqual(q.intrinsics, p.intrinsics)

    def test_bad_resolution_identifies_role(self):
        a = scene(); a['background_depth'] = np.ones((480,640))
        with self.assertRaisesRegex(ValueError, 'background_depth: input must'):
            prepare_inputs(a)

    def test_worker_loads_mixed_files_same_as_pre_normalized(self):
        from depth_image_lens.tuner import execute_job
        a = scene(); a['color'] = large(a['color']); a['background_color'] = large(a['background_color'])
        c = default_config(); c['diagnostics'] = True
        with tempfile.TemporaryDirectory() as td:
            paths={}
            for key, value in a.items():
                p=Path(td)/(key+'.npy'); np.save(p,value); paths[key]=str(p)
            got=execute_job(dict(kind='preview',paths=paths,crop=[40,40,80,90],config=c))
        expected=run_pipeline(prepare_inputs(scene(),[40,40,80,90]),c)
        np.testing.assert_array_equal(got.object_mask,expected.object_mask)
        np.testing.assert_array_equal(got.arrays['segmenter/depth_difference_m'],expected.arrays['segmenter/depth_difference_m'])


if __name__ == '__main__': unittest.main()
