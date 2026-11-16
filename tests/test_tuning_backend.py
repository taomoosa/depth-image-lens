"""Local tuning input contracts and synthetic workflow tests."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from depth_image_lens.tuning_backend import (default_config, parameter_defaults, load_array,
    prepare_inputs, run_pipeline, evaluate_dataset, search_parameters, load_config,
    save_config, load_manifest)
from depth_image_lens.geometry import CameraIntrinsics


def scene():
    color = np.full((400,640,3), 100, np.uint8)
    background = color.copy()
    mask = np.zeros((400,640), bool)
    mask[50:100,50:100] = True
    color[mask] = [200,20,20]
    depth = np.full((400,640),1000,np.uint16)
    depth[mask] = 980
    return dict(color=color, background_color=background, depth=depth,
                background_depth=np.full_like(depth,1000), mask=mask,
                ground_truth_mask=mask.copy())


class TuningBackendTests(unittest.TestCase):
    def test_defaults_json(self):
        c = default_config()
        self.assertEqual(c['segmenter']['fusion_mode'], 'union')
        self.assertNotIn('background_segmenter', c['counter'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            save_config(path,c)
            self.assertEqual(load_config(path),c)

    def test_resize_crop_no_depth_interpolation(self):
        a = scene()
        big = {k: np.repeat(np.repeat(v,2,axis=0),2,axis=1) for k,v in a.items()}
        big['depth'][100,100] = 0
        p = prepare_inputs(big,[40,40,80,90],CameraIntrinsics(800,800,639.5,399.5))
        self.assertEqual(p.arrays['depth'].shape,(90,80))
        self.assertEqual(p.arrays['depth'][10,10],0)
        self.assertEqual(p.arrays['depth'][11,11],980)
        self.assertEqual(p.intrinsics.fx,400)
        self.assertEqual(p.intrinsics.cx,279.5)
        np.testing.assert_allclose(p.to_source_xy([[0,0]]),[[80.5,80.5]])
        self.assertEqual(p.arrays['mask'].dtype,bool)

    def test_invalid_depth_preserved(self):
        z = np.full((800,1280),np.nan)
        z[0,0] = np.inf
        p = prepare_inputs({'depth':z},[0,0,2,2])
        self.assertTrue(np.isinf(p.arrays['depth'][0,0]))
        self.assertTrue(np.isnan(p.arrays['depth'][1,1]))

    def test_bad_grid_crop(self):
        for crop in ([0,0,641,400],[-1,0,3,3],[0,0,1,4],[0.,0,5,5]):
            with self.assertRaises(ValueError): prepare_inputs(scene(),crop)
        with self.assertRaises(ValueError): prepare_inputs({'color':np.zeros((5,5,3),np.uint8)})
        a = scene(); a['mask'] = a['mask'][:20]
        with self.assertRaises(ValueError): prepare_inputs(a)

    def test_input_snapshots(self):
        a=scene(); p=prepare_inputs(a,[40,40,80,80]); p.arrays['color'][:]=0
        self.assertTrue(a['color'].any())

    def test_safe_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            np.save(p/'bad.npy',np.array([{'x':1}],object))
            with self.assertRaises(ValueError): load_array(p/'bad.npy','depth')
            np.savez(p/'z.npz',x=np.ones((2,2)),y=np.zeros((2,2)))
            with self.assertRaises(ValueError): load_array(p/'z.npz','depth')
            np.testing.assert_equal(load_array(p/'z.npz','depth','x'),1)
            with self.assertRaises(ValueError): load_array(p/'z.jpg','depth')

    def test_mask_pipeline_and_outline(self):
        p=prepare_inputs(scene(),[40,40,80,80]); c=default_config(); c['source']='mask'
        r=run_pipeline(p,c)
        np.testing.assert_array_equal(r.object_mask,p.arrays['mask'])
        self.assertIn('result/overlay',r.images)
        c['target']='outline'; r=run_pipeline(p,c)
        self.assertIn('outline',r.summary['stages'])
        self.assertIn('outline/debug',r.images)

    def test_segmentation_modes_and_diagnostics_invariance(self):
        from depth_image_lens.tuning_backend import MODES
        p=prepare_inputs(scene(),[40,40,80,80]); c=default_config()
        for mode in MODES:
            c['segmenter']['fusion_mode']=mode
            c['diagnostics']=False; baseline=run_pipeline(p,c)
            c['diagnostics']=True; detailed=run_pipeline(p,c)
            np.testing.assert_array_equal(baseline.object_mask,detailed.object_mask,err_msg=mode)
            self.assertIn('segmenter/depth_difference_m', detailed.images)

    def test_counter_mask_background(self):
        p=prepare_inputs(scene(),[40,40,80,80]); c=default_config(); c['source']='mask'; c['target']='count'
        r=run_pipeline(p,c)
        self.assertEqual(r.summary['stages']['counter']['candidate_count'],1)
        self.assertIn('counter/debug',r.images)

    def test_repair_changes_propagated(self):
        a=scene(); a['mask'][70,70]=False
        p=prepare_inputs(a,[40,40,80,80]); c=default_config(); c['source']='mask'; c['repair_enabled']=True
        c['repair']['max_hole_area']=1
        r=run_pipeline(p,c)
        self.assertTrue(r.object_mask[30,30])
        self.assertTrue(r.arrays['repair/added_mask'][30,30])

    def test_search_disabled_stage(self):
        p=prepare_inputs(scene(),[40,40,80,80]); c=default_config(); c['source']='mask'
        with self.assertRaisesRegex(ValueError, 'disabled'):
            search_parameters([dict(id='one',prepared=p)],c,{'repair.max_hole_area':[0,1]})

    def test_search_split_and_labels(self):
        p=prepare_inputs(scene(),[40,40,80,80]); c=default_config(); c['source']='mask'
        c['repair_enabled']=True
        s=[dict(id='one',prepared=p)]
        rows=evaluate_dataset(s,c)
        self.assertEqual(rows[0]['metrics']['mask_iou'],1)
        result=search_parameters(s,c,{'repair.max_hole_area':[0,1]},validation_samples=[dict(id='two',prepared=p)])
        self.assertEqual(len(result['trials']),2)
        self.assertEqual(result['best']['score'],1)
        self.assertFalse(result['validation_used_for_selection'])
        with self.assertRaises(ValueError): search_parameters(s,c,{'repair.max_hole_area':[0]},validation_samples=s)
        q=prepare_inputs({'color':scene()['color'],'mask':scene()['mask']},[40,40,80,80])
        with self.assertRaises(ValueError): search_parameters([dict(id='x',prepared=q)],c,{'repair.max_hole_area':[0]})

    def test_search_random_reproducible_cancel(self):
        p=prepare_inputs(scene(),[40,40,80,80]); c=default_config(); c['source']='mask'
        c['repair_enabled']=True
        s=[dict(id='one',prepared=p)]
        args=dict(method='random',seed=12,max_trials=3)
        space={'repair.max_hole_area':list(range(10))}
        a=search_parameters(s,c,space,**args); b=search_parameters(s,c,space,**args)
        self.assertEqual([r['parameters'] for r in a['trials']],[r['parameters'] for r in b['trials']])
        self.assertEqual(search_parameters(s,c,space,cancel=lambda:True)['trials'],[])

    def test_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); np.save(p/'color.npy',scene()['color'])
            (p/'dataset.json').write_text(json.dumps(dict(version=1,samples=[dict(id='a',arrays={'color':'color.npy'},split='validation',expected_count=0)])))
            rows=load_manifest(p/'dataset.json')
            self.assertEqual(rows[0]['split'],'validation')
            self.assertEqual(rows[0]['arrays']['color'].shape,(400,640,3))


if __name__ == '__main__': unittest.main()
