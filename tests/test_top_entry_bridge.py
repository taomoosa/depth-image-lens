import json
import unittest
import numpy as np
from depth_image_lens import TopEntryBridgeDetector


def scene(bridge=True, left=12, right=55, thickness=7):
    mask = np.zeros((60, 90), bool)
    mask[:30, left:left+9] = True
    mask[:30, right:right+9] = True
    if bridge:
        mask[23:23+thickness, left:right+9] = True
    depth = np.where(mask, 1., 1.5)
    return depth, mask


class TopEntryBridgeTests(unittest.TestCase):
    def detect(self, z, mask, **kwargs):
        return TopEntryBridgeDetector().detect_mask(z, mask, depth_unit='m', diagnostics=True, **kwargs)

    def test_bridge_and_serialization(self):
        z, m = scene()
        r = self.detect(z, m)
        self.assertEqual(r.bridge, 'yes')
        self.assertTrue(r.candidate_bridge)
        self.assertEqual(len(r.entries), 2)
        self.assertGreater(r.metrics['robust_core_pixels'], 0)
        json.dumps(r.to_dict(), allow_nan=False)
        self.assertNotIn('diagnostics', r.to_dict())

    def test_empty_and_one_and_three_entries_are_unknown(self):
        z, m = scene()
        for variant in [np.zeros_like(m), m & (np.indices(m.shape)[1] < 40)]:
            self.assertEqual(self.detect(z, variant).bridge, 'unknown')
        m[:20, 36:43] = True
        self.assertEqual(self.detect(z, m).bridge, 'unknown')

    def test_separate_requires_completeness(self):
        z, m = scene(False)
        self.assertEqual(self.detect(z, m).bridge, 'unknown')
        r = self.detect(z, m, mask_complete=True)
        self.assertEqual(r.bridge, 'no')
        self.assertFalse(r.candidate_bridge)

    def test_close_but_distinct_entries(self):
        z, m = scene(left=30, right=41)
        self.assertEqual(self.detect(z, m).bridge, 'yes')
        z, m = scene(left=30, right=40)
        self.assertEqual(self.detect(z, m).bridge, 'unknown')
        z, m = scene(left=30, right=38)
        self.assertEqual(self.detect(z, m).bridge, 'unknown')

    def test_top_join_not_bridge(self):
        z, m = scene(False)
        m[2:5, 12:64] = True
        self.assertEqual(self.detect(z, m).bridge, 'unknown')

    def test_tiny_specks_ignored(self):
        z, m = scene()
        m[0, 40] = True; m[48:50, 70:72] = True
        z[m] = 1.
        self.assertEqual(self.detect(z, m).bridge, 'yes')

    def test_isolated_boundary_speck_does_not_veto_no(self):
        z, m = scene(False)
        m[59, 40] = True; z[59, 40] = 1.
        self.assertEqual(self.detect(z, m, mask_complete=True).bridge, 'no')

    def test_thin_false_connector_is_unknown(self):
        z, m = scene(thickness=1)
        r = self.detect(z, m, mask_complete=True)
        self.assertEqual(r.bridge, 'unknown')
        self.assertTrue(r.candidate_bridge)
        self.assertIn('observed_connection_too_thin', r.reasons)

    def test_other_depth_surface_cannot_supply_path_width(self):
        z, m = scene()
        z[23:30, 21:55] = 1.3
        z[26, 21:55] = 1.0
        r = self.detect(z, m)
        self.assertEqual(r.bridge, 'unknown')
        self.assertTrue(r.candidate_bridge)
        self.assertIn('observed_connection_too_thin', r.reasons)

    def test_depth_spike_has_alternative_path(self):
        z, m = scene()
        z[26, 40] = .2
        self.assertEqual(self.detect(z, m).bridge, 'yes')

    def test_small_observed_hole_can_be_bypassed(self):
        z, m = scene(thickness=12)
        m[26, 40] = False
        self.assertEqual(self.detect(z, m).bridge, 'yes')

    def test_cut_mask_candidate_not_yes(self):
        z, m = scene()
        m[:, 38] = False
        r = self.detect(z, m, mask_complete=True)
        self.assertEqual(r.bridge, 'unknown')
        self.assertTrue(r.candidate_bridge)
        self.assertGreater(r.metrics['inferred_pixels'], 0)

    def test_invalid_depth_cut_is_never_filled_as_evidence(self):
        z, m = scene()
        z[:, 38] = np.nan
        r = self.detect(z, m, mask_complete=True)
        self.assertEqual(r.bridge, 'unknown')
        self.assertTrue(r.candidate_bridge)
        self.assertFalse(r.diagnostics['observed_path_mask'].any())

    def test_depth_jump_splits_geometric_path(self):
        z, m = scene()
        z[:, 38:] = np.where(m[:, 38:], 1.2, 1.5)
        r = self.detect(z, m, mask_complete=True)
        self.assertEqual(r.bridge, 'unknown')
        self.assertGreater(r.metrics['rejected_depth_edges'], 0)

    def test_uncertain_repair_does_not_prove_bridge(self):
        z, m = scene()
        uncertain = np.zeros_like(m); uncertain[:, 38] = True
        r = self.detect(z, m, uncertain_mask=uncertain)
        self.assertEqual(r.bridge, 'unknown')
        self.assertTrue(r.candidate_bridge)

    def test_exclusion_cannot_be_repaired(self):
        z, m = scene()
        excluded = np.zeros_like(m); excluded[:, 38] = True
        r = self.detect(z, m, exclude_mask=excluded, mask_complete=True)
        self.assertEqual(r.bridge, 'unknown')
        self.assertFalse(r.candidate_bridge)

    def test_missing_background_with_no_bridge_is_unknown(self):
        z, m = scene(False)
        z[40, 40] = 0
        self.assertEqual(self.detect(z, m, mask_complete=True).bridge, 'unknown')

    def test_bottom_clipping_prevents_no(self):
        z, m = scene(False)
        m[30:, 12:21] = True; z[m] = 1.
        self.assertEqual(self.detect(z, m, mask_complete=True).bridge, 'unknown')

    def test_mm_equivalence_and_no_input_mutation(self):
        z, m = scene(); original_z = z.copy(); original_m = m.copy()
        r = TopEntryBridgeDetector().detect_mask(z*1000, m, depth_unit='mm')
        self.assertEqual(r.bridge, self.detect(z, m).bridge)
        np.testing.assert_array_equal(z, original_z)
        np.testing.assert_array_equal(m, original_m)

    def test_parameters_and_inputs(self):
        for options in [{'min_path_width_px':2}, {'entry_band_height_px':2},
                        {'gap_radius_px':4}, {'depth_jump_m':0}, {'min_area':True},
                        {'min_entry_row_fraction':float('nan')}]:
            with self.assertRaises(ValueError): TopEntryBridgeDetector(**options)
        z, m = scene()
        for options in [{'depth_unit':'auto'}, {'depth_unit':'m', 'mask_complete':1}]:
            with self.assertRaises(ValueError): TopEntryBridgeDetector().detect_mask(z, m, **options)
        with self.assertRaises(ValueError): self.detect(z, m.astype(np.uint8))


if __name__ == '__main__':
    unittest.main()
