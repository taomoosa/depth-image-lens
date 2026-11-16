"""Opt-in diagnostics expose independent gates without changing decisions."""
import unittest
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter
from depth_image_lens._rgbd_debug import growth_boundary_diagnostics
from .test_rgbd_background import scene, detect

MODES = ('union','depth_required','evidence','color_seeded','depth_anchored',
         'color_supported','color_region','color_local','color_model')

class DetailedDebugTests(unittest.TestCase):
    def test_all_modes_on_off_identical_selection_and_uncertainty(self):
        c,z,bc,bz=scene()
        c[5:30,8:40]=[190,35,40]; z[5:30,8:40]=997
        z[10:15,10:15]=0; z[18:20,10:20]=1005
        for mode in MODES:
            with self.subTest(mode=mode):
                d=RGBDBackgroundSegmenter(fusion_mode=mode)
                off=detect(c,z,bc,bz,d,diagnostics=False)
                on=detect(c,z,bc,bz,d,diagnostics=True)
                self.assertFalse(off.diagnostics)
                for key,value in vars(off).items():
                    if isinstance(value,np.ndarray):
                        np.testing.assert_array_equal(value,getattr(on,key))
                self.assertEqual(off.stages,on.stages)
                self.assertEqual(off.metrics,on.metrics)
                self.assertIn('diagnostic_metadata',on.diagnostics)
                self.assertTrue(on.diagnostics['reason_depth_unavailable_mask'][11,11])
                self.assertEqual(on.diagnostics['gate_depth_foreground_margin'].shape,z.shape)
    def test_local_failed_gates_overlap_and_scope(self):
        c,z,bc,bz=scene(); c[5:30,8:40]=[190,35,40]
        r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='color_local'),diagnostics=True)
        d=r.diagnostics
        self.assertTrue(d['reason_local_mean_failed_mask'][15,20])
        self.assertTrue(d['reason_local_density_failed_mask'][15,20])
        self.assertTrue(d['reason_locally_table_like_mask'][15,20])
        self.assertTrue(np.isnan(d['gate_local_mean_depth_margin'][0,0]))
    def test_growth_frontier_color_and_budget_are_separate(self):
        lab=np.zeros((5,7,3)); proto=lab.copy()
        selected=np.zeros((5,7),bool); selected[2,2]=True
        domain=np.ones((5,7),bool); lab[2,3]=100
        d=growth_boundary_diagnostics(lab,proto,domain,selected,5.)
        self.assertTrue(d['reason_growth_color_failed_mask'][2,3])
        self.assertTrue(d['reason_growth_compatible_next_step_mask'][2,1])
        self.assertTrue(d['reason_growth_no_selected_neighbor_mask'][0,6])
        self.assertFalse(d['reason_growth_compatible_next_step_mask'][2,3])
        self.assertFalse(d['reason_growth_color_failed_mask'][2,2])
    def test_min_area_reason_survives_final_decision(self):
        c,z,bc,bz=scene(); z[3,3]=970
        r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode='depth_required',min_object_area=5),diagnostics=True)
        self.assertTrue(r.diagnostics['reason_removed_by_min_object_area_mask'][3,3])
        self.assertFalse(r.diagnostics['fusion_final_selected_mask'][3,3])

    def test_final_area_filter_is_not_a_seed_area_failure(self):
        c,z,bc,bz=scene(); c[5:30,8:40]=[190,35,40]; z[5:30,8:40]=970
        for mode, parameter, reason in (
                ('color_local','local_seed_min_area','reason_local_seed_area_filtered_mask'),
                ('color_supported','supported_min_seed_area','reason_seed_area_filtered_mask')):
            with self.subTest(mode=mode):
                r=detect(c,z,bc,bz,RGBDBackgroundSegmenter(fusion_mode=mode,min_object_area=10000,**{parameter:1}),diagnostics=True)
                self.assertTrue(r.diagnostics['filtered_mask'].any())
                self.assertFalse(r.diagnostics[reason].any())
