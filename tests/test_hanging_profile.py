import json
import unittest
import numpy as np
from depth_image_lens import HangingProfileAnalyzer


class HangingProfileTests(unittest.TestCase):
    def scene(self):
        mask = np.zeros((80, 100), bool)
        mask[10:60, 20:80] = True
        return mask

    def run_profile(self, mask, **kw):
        return HangingProfileAnalyzer(reach_drop_px=40, upper_band_height_px=10).analyze_mask(
            mask, reference_y=10, expected_x_range=(15, 85), **kw)

    def test_rectangle_exact_and_arrays(self):
        r = self.run_profile(self.scene(), mask_complete=True)
        self.assertEqual(r.metrics['drop_quantiles_px']['p50'], 49)
        self.assertEqual(r.metrics['exact_drop_quantiles_px']['p10'], 49)
        self.assertAlmostEqual(r.metrics['upper_area_fraction'], .2)
        self.assertAlmostEqual(r.metrics['reaching_width_fraction'], 60/70)
        np.testing.assert_array_equal(r.profiles['row_width_px'][10:60], 60)
        self.assertTrue(np.isnan(r.profiles['bottom_y_px'][:20]).all())
        self.assertEqual(r.metrics['classification'], 'not_performed')

    def test_default_not_complete(self):
        r = self.run_profile(self.scene())
        self.assertEqual(r.metrics['exact_columns'], 0)
        self.assertEqual(r.metrics['censored_columns'], 60)
        self.assertIsNone(r.metrics['exact_drop_quantiles_px']['p50'])
        self.assertEqual(r.metrics['drop_quantiles_px']['p50'], 49)

    def test_empty_json_and_no_zero_fake_bottom(self):
        r = self.run_profile(np.zeros((80, 100), bool), mask_complete=True)
        self.assertIsNone(r.metrics['drop_quantiles_px']['p50'])
        self.assertTrue(np.isnan(r.profiles['bottom_y_px']).all())
        self.assertEqual(r.metrics['reaching_width_fraction'], 0)
        json.dumps(r.to_dict(include_profiles=True), allow_nan=False)

    def test_missing_below_censors_not_zero(self):
        m = self.scene(); observed = np.ones_like(m)
        observed[65:70, 30:40] = False
        r = self.run_profile(m, observed_mask=observed, mask_complete=True)
        self.assertEqual(r.metrics['censored_columns'], 10)
        np.testing.assert_array_equal(r.profiles['bottom_y_px'][30:40], 59)
        self.assertTrue(np.isnan(r.profiles['exact_drop_px'][30:40]).all())

    def test_unknown_empty_column_prevents_complete_distribution(self):
        m=self.scene(); obs=np.ones_like(m);obs[:,16]=False
        r=self.run_profile(m,observed_mask=obs,mask_complete=True)
        self.assertEqual(r.metrics['exact_columns'],60)
        self.assertFalse(r.metrics['complete_bottom_distribution'])

    def test_uncertain_repairs_not_observed(self):
        m = self.scene(); m[60:70, 30:40] = True
        uncertain = np.zeros_like(m); uncertain[60:70, 30:40] = True
        r = self.run_profile(m, uncertain_mask=uncertain, mask_complete=True, diagnostics=True)
        self.assertEqual(r.metrics['inferred_foreground_pixels'], 100)
        self.assertEqual(r.metrics['censored_columns'], 10)
        np.testing.assert_array_equal(r.profiles['bottom_y_px'][30:40], 59)
        self.assertFalse(r.diagnostics['retained_observed_mask'][60:70].any())

    def test_bottom_crop_lower_bound(self):
        m = self.scene(); m[60:, 20:80] = True
        r = self.run_profile(m, mask_complete=True)
        self.assertEqual(r.metrics['drop_quantiles_px']['p50'], 69)
        self.assertEqual(r.metrics['censored_columns'], 60)
        self.assertTrue(r.metrics['boundary_touches']['bottom'])
        self.assertFalse(r.metrics['complete_bottom_distribution'])

    def test_noise_short_runs_and_component_filter(self):
        m = self.scene(); m[75, 30:70] = True; m[70, 21] = True
        r = self.run_profile(m, mask_complete=True)
        self.assertEqual(r.profiles['raw_bottom_y_px'][21], 70)
        self.assertEqual(r.profiles['raw_bottom_y_px'][40], 75)
        self.assertEqual(r.profiles['bottom_y_px'][40], 59)
        self.assertEqual(r.profiles['bottom_y_px'][21], 59)
        self.assertEqual(r.metrics['removed_small_component_pixels'], 1)
        self.assertEqual(r.metrics['unsupported_profile_pixels'], 40)

    def test_gap_not_interpolated_and_fragments(self):
        m = self.scene(); m[:, 45:55] = False
        r = self.run_profile(m, mask_complete=True)
        self.assertTrue(np.isnan(r.profiles['smoothed_bottom_y_px'][45:55]).all())
        self.assertEqual(r.metrics['retained_component_count'], 2)
        self.assertIn('multiple_fragments_not_automatically_associated', r.warnings)

    def test_tail_reference_width_not_only_occupied_width(self):
        m = np.zeros((80, 100), bool); m[10:70, 48:52] = True
        r = self.run_profile(m, mask_complete=True)
        self.assertEqual(r.metrics['reaching_observed_columns_fraction'], 1)
        self.assertAlmostEqual(r.metrics['reaching_width_fraction'], 4/70)

    def test_same_mean_different_quantiles(self):
        a = np.zeros((120,100), bool); b = a.copy()
        a[10:61,20:80] = True
        b[10:41,20:60] = True; b[10:101,60:80] = True
        ra=self.run_profile(a,mask_complete=True); rb=self.run_profile(b,mask_complete=True)
        self.assertEqual(ra.metrics['drop_mean_px'], rb.metrics['drop_mean_px'])
        self.assertEqual(ra.metrics['drop_mean_px'], 50)
        self.assertNotEqual(ra.metrics['drop_quantiles_px']['p50'],rb.metrics['drop_quantiles_px']['p50'])
        self.assertGreater(rb.metrics['drop_iqr_px'],ra.metrics['drop_iqr_px'])

    def test_discarded_lower_observation_prevents_exact_claim(self):
        m = np.zeros((20,10), bool); m[2:10,2:8]=True; m[15:17,4:6]=True
        r = HangingProfileAnalyzer(min_component_area=1).analyze_mask(
            m, reference_y=0, expected_x_range=(2,8), mask_complete=True)
        self.assertEqual(r.profiles['bottom_y_px'][4], 9)
        self.assertEqual(r.profiles['raw_bottom_y_px'][4], 16)
        self.assertTrue(r.profiles['bottom_censored'][4])
        self.assertFalse(r.metrics['complete_bottom_distribution'])
        self.assertEqual(r.metrics['exact_columns'], 4)

    def test_inputs_not_mutated_and_debug_on_off(self):
        m=self.scene(); original=m.copy(); obs=np.ones_like(m); unc=np.zeros_like(m)
        a=self.run_profile(m,observed_mask=obs,uncertain_mask=unc,diagnostics=False)
        b=self.run_profile(m,observed_mask=obs,uncertain_mask=unc,diagnostics=True)
        self.assertEqual(a.metrics,b.metrics)
        for k in a.profiles:
            np.testing.assert_array_equal(a.profiles[k],b.profiles[k])
        np.testing.assert_array_equal(m,original)
        self.assertEqual(a.diagnostics,{})
        with self.assertRaises(ValueError): a.render_debug()

    def test_negative_ref_and_fractional_ref(self):
        m=self.scene()
        r=HangingProfileAnalyzer().analyze_mask(m,reference_y=-5,mask_complete=True)
        self.assertEqual(r.metrics['drop_quantiles_px']['p50'],64)
        r=HangingProfileAnalyzer().analyze_mask(m,reference_y=10.5,mask_complete=True)
        self.assertEqual(r.metrics['drop_quantiles_px']['p50'],48.5)

    def test_validation(self):
        for kwargs in [dict(min_component_area=0),dict(min_vertical_run_px=True),
                       dict(smoothing_window_px=2),dict(quantiles=[]),dict(quantiles=[.5,.5]),
                       dict(quantiles=[1.1]),dict(reach_drop_px=float('nan'))]:
            with self.assertRaises(ValueError): HangingProfileAnalyzer(**kwargs)
        a=HangingProfileAnalyzer();m=self.scene()
        for kwargs in [dict(reference_y=80),dict(reference_y=float('inf')),
                       dict(reference_y=10,expected_x_range=(3,3)),
                       dict(reference_y=10,observed_mask=np.ones((1,1),bool)),
                       dict(reference_y=10,mask_complete=1)]:
            with self.assertRaises(ValueError): a.analyze_mask(m,**kwargs)
        with self.assertRaises(ValueError): a.analyze_mask(m.astype(np.uint8),reference_y=0)

    def test_close_quantiles_do_not_overwrite(self):
        r=HangingProfileAnalyzer(quantiles=(.5,.500000001)).analyze_mask(self.scene(), reference_y=10)
        self.assertEqual(len(r.metrics['drop_quantiles_px']),2)

    def test_debug_rgb_shape(self):
        r=self.run_profile(self.scene(),diagnostics=True)
        try:
            image=r.render_debug()
        except ImportError:
            self.skipTest('OpenCV optional')
        self.assertEqual(image.shape,(180,720,3))
        self.assertEqual(image.dtype,np.uint8)

if __name__=='__main__': unittest.main()
