"""Direct mask evidence contracts; synthetic fixtures are not accuracy validation."""
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from depth_image_lens import OverheadObjectCounter, RGBDBackgroundSegmenter, cleanup_mask


def scene():
    color = np.full((60, 80, 3), 150, np.uint8)
    depth = np.full((60, 80), 1000, np.uint16)
    empty = depth.copy()
    mask = np.zeros(depth.shape, bool)
    mask[10:30, 10:30] = True
    color[mask] = (210, 35, 35)
    depth[mask] = 980
    return color, mask, depth, empty


def run(arrays, detector=None, **options):
    color, mask, depth, empty = arrays
    args = dict(color_order='RGB', depth=depth, background_depth=empty,
                depth_scale_m=.001, registered_to_depth=True)
    return (detector or OverheadObjectCounter()).detect_mask(color, mask, **dict(args, **options))


class DirectMaskCountTests(unittest.TestCase):
    def test_direct_mode_does_not_call_segmenter_or_need_empty_color(self):
        segmenter = RGBDBackgroundSegmenter(fusion_mode='color_seeded', min_object_area=1000)
        with patch.object(segmenter, 'detect', side_effect=AssertionError('must not segment')):
            result = run(scene(), OverheadObjectCounter(background_segmenter=segmenter), diagnostics=True)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.parameters['input_mode'], 'external_mask')
        self.assertIsNone(result.parameters['background_segmenter'])
        self.assertIsNone(result.diagnostics['background'])
        self.assertNotIn('color_foreground_mask', result.diagnostics)
        self.assertEqual(result.metrics['color_reference_status'], 'not_used_external_mask')

    def test_raw_segmentation_mask_matches_original_path_on_supported_scene(self):
        color, mask, depth, empty = arrays = scene()
        mask[10:30, 30:50] = True
        color[10:30, 30:50] = (35, 35, 210)
        depth[10:30, 30:50] = 970
        counter = OverheadObjectCounter()
        original = counter.detect(color, depth, background_color=np.full_like(color, 150),
                                  background_depth=empty, color_order='RGB',
                                  depth_scale_m=.001, registered_to_depth=True)
        direct = run(arrays, counter)
        self.assertEqual(direct.count, 2)
        self.assertEqual(direct.groups, original.groups)
        np.testing.assert_array_equal(direct.label_map, original.label_map)
        np.testing.assert_array_equal(direct.observed_mask, original.observed_mask)

    def test_mask_without_depth_returns_hypotheses_and_explicit_abstention(self):
        color, mask, _, _ = scene()
        result = OverheadObjectCounter().detect_mask(color, mask, color_order='RGB')
        self.assertEqual(result.candidate_count, 1)
        self.assertIsNone(result.count)
        self.assertEqual(result.state, 'insufficient')
        self.assertIn('external_mask_has_no_depth_evidence', result.reasons)
        self.assertFalse(result.observed_mask.any())
        self.assertTrue(result.uncertain_mask.all())
        self.assertIsNone(result.parameters['depth_scale_m'])
        json.dumps(result.to_dict(), allow_nan=False)

    def test_empty_mask_needs_observed_depth(self):
        arrays = color, mask, depth, _ = scene()
        mask[:] = False
        depth[:] = 1000
        result = run(arrays)
        self.assertEqual((result.count, result.state), (0, 'no_visible_candidates'))
        result = OverheadObjectCounter().detect_mask(color, mask, color_order='RGB')
        self.assertIsNone(result.count)
        self.assertEqual(result.candidate_count, 0)

    def test_depth_threshold_and_margin_use_existing_configuration(self):
        arrays = scene()
        counter = OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(
            depth_threshold_m=.03, depth_margin_m=.002))
        result = run(arrays, counter)
        self.assertIsNone(result.count)
        self.assertEqual(result.groups[0]['depth_supported_fraction'], 0.)
        self.assertEqual(result.parameters['depth_support_parameters'],
                         dict(depth_threshold_m=.03, depth_margin_m=.002))
        # Six stored millimetres is foreground but within the configured margin.
        arrays[2][arrays[1]] = 994
        counter = OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(
            depth_margin_m=.0015))
        result = run(arrays, counter)
        self.assertIsNone(result.count)
        self.assertEqual(result.groups[0]['depth_supported_fraction'], 0.)

    def test_false_current_or_reference_validity_is_not_observed_empty(self):
        for key in ('depth_valid_mask', 'background_depth_valid_mask', 'color_valid_mask', 'observed_mask'):
            arrays = scene()
            valid = np.ones(arrays[1].shape, bool)
            valid[0, 0] = False
            result = run(arrays, **{key: valid})
            self.assertIsNone(result.count)
            self.assertFalse(result.observed_mask[0, 0])
            self.assertTrue(result.uncertain_mask[0, 0])
            self.assertIn('incomplete_rgbd_observation', result.reasons)

    def test_unobserved_foreground_cannot_be_grouped(self):
        arrays = scene()
        observed = ~arrays[1]
        result = run(arrays, observed_mask=observed)
        self.assertEqual(result.candidate_count, 0)
        self.assertTrue(result.rejected_mask[arrays[1]].all())
        self.assertFalse(result.object_mask.any())

    def test_missing_depth_and_sentinels_preserve_unknown(self):
        for target in (2, 3):
            for value in (0., -1., np.nan, np.inf, 65535.):
                arrays = list(scene())
                arrays[target] = arrays[target].astype(float)
                arrays[target][0, 0] = value
                result = run(arrays, invalid_values=(65535,))
                self.assertIsNone(result.count)
                self.assertFalse(result.observed_mask[0, 0])
                self.assertTrue(result.uncertain_mask[0, 0])

    def test_depth_scale_overflow_and_underflow_do_not_create_evidence(self):
        for scale in (1e308, 5e-324):
            arrays = list(scene())
            if scale < 1:
                arrays[2] = np.full(arrays[1].shape, .1)
                arrays[3] = np.full(arrays[1].shape, .2)
            result = run(arrays, depth_scale_m=scale)
            self.assertIsNone(result.count)
            self.assertFalse(result.observed_mask.any())

    def test_farther_depth_is_conflict_not_unsigned_wraparound(self):
        arrays = scene()
        arrays[2][arrays[1]] = 1020
        result = run(arrays, diagnostics=True)
        self.assertIsNone(result.count)
        self.assertIn('color_depth_conflict', result.reasons)
        self.assertTrue(np.all(result.diagnostics['depth_difference_m'][arrays[1]] < 0))
        self.assertEqual(result.groups[0]['depth_supported_fraction'], 0.)

    def test_shadow_fragment_cannot_borrow_support(self):
        arrays = color, mask, _, _ = scene()
        mask[35:55, 50:70] = True
        color[35:55, 50:70] = (210, 35, 35)
        result = run(arrays)
        self.assertEqual(result.candidate_count, 1)
        self.assertIsNone(result.count)
        self.assertIn('fragment_without_depth_support', result.reasons)
        self.assertEqual(result.groups[0]['unsupported_fragment_pixels'], 400)

    def test_explicit_input_uncertainty_retained_even_outside_object(self):
        for where in ((15, 15), (0, 0)):
            arrays = scene()
            uncertain = np.zeros(arrays[1].shape, bool)
            uncertain[where] = True
            result = run(arrays, uncertain_mask=uncertain, diagnostics=True)
            self.assertIsNone(result.count)
            self.assertTrue(result.uncertain_mask[where])
            self.assertIn('external_mask_has_uncertain_pixels', result.reasons)
            self.assertEqual(result.metrics['input_uncertain_pixels'], 1)
            self.assertFalse(result.diagnostics['depth_foreground_mask'][where])

    def test_cleanup_additions_and_removals_are_not_measured_support(self):
        arrays = color, mask, _, _ = scene()
        raw = mask.copy()
        raw[15:17, 15:17] = False
        repair = cleanup_mask(raw, max_hole_area=4)
        self.assertEqual(repair.changed_mask.sum(), 4)
        result = run((color, repair.object_mask, arrays[2], arrays[3]),
                     cleanup_changed_mask=repair.changed_mask, diagnostics=True)
        self.assertEqual(result.candidate_count, 1)
        self.assertIsNone(result.count)
        self.assertEqual(result.metrics['cleanup_changed_pixels'], 4)
        self.assertTrue(result.uncertain_mask[repair.changed_mask].all())
        self.assertFalse(result.diagnostics['depth_foreground_mask'][repair.changed_mask].any())
        self.assertAlmostEqual(result.groups[0]['depth_supported_fraction'], .99)
        # Removed pixels must remain visible even though absent from the new mask.
        removed = np.zeros(mask.shape, bool)
        removed[15:17, 15:17] = True
        result = run((color, raw, arrays[2], arrays[3]), cleanup_changed_mask=removed)
        self.assertIsNone(result.count)
        self.assertTrue(result.uncertain_mask[removed].all())
        self.assertFalse(result.candidate_mask[removed].any())

    def test_noop_cleanup_does_not_change_count(self):
        arrays = scene()
        repair = cleanup_mask(arrays[1])
        result = run(arrays, cleanup_changed_mask=repair.changed_mask)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.metrics['cleanup_changed_pixels'], 0)

    def test_exclusions_and_roi_limit_uncertainty_without_labeling_it_empty(self):
        arrays = scene()
        uncertain = np.zeros(arrays[1].shape, bool)
        uncertain[0, 0] = True
        result = run(arrays, exclusion_mask=uncertain, uncertain_mask=uncertain,
                     cleanup_changed_mask=uncertain)
        self.assertEqual(result.count, 1)
        self.assertTrue(result.excluded_mask[0, 0])
        self.assertFalse(result.observed_mask[0, 0])
        roi = ~uncertain
        result = run(arrays, work_roi=roi, uncertain_mask=uncertain, cleanup_changed_mask=uncertain)
        self.assertEqual(result.count, 1)
        self.assertFalse(result.observed_mask[0, 0])
        self.assertFalse(result.uncertain_mask[0, 0])

    def test_empty_domain_never_claims_zero(self):
        arrays = scene()
        for options in (dict(work_roi=np.zeros(arrays[1].shape, bool)),
                        dict(exclusion_mask=np.ones(arrays[1].shape, bool))):
            result = run(arrays, **options)
            self.assertIsNone(result.count)
            self.assertEqual(result.state, 'insufficient')

    def test_arm_guard_and_upper_depth_band_use_same_common_path(self):
        arrays = color, mask, depth, _ = scene()
        mask[10:50, 50:70] = True
        color[10:50, 50:70] = (200, 140, 100)
        depth[10:50, 50:70] = 700
        arm = np.zeros(mask.shape, bool)
        arm[11:49, 51:69] = True
        for detector, options in ((OverheadObjectCounter(), dict(arm_mask=arm)),
                                  (OverheadObjectCounter(max_reference_separation_m=.15), {})):
            result = run(arrays, detector, **options)
            self.assertEqual(result.count, 1)
            self.assertTrue(result.excluded_mask[10:50, 50:70].all())

    def test_missing_depth_cannot_pass_upper_band(self):
        arrays = scene()
        arrays[2][arrays[1]] = 0
        result = run(arrays, OverheadObjectCounter(max_reference_separation_m=.15))
        self.assertIsNone(result.count)
        self.assertEqual(result.candidate_count, 0)
        self.assertIn('missing_depth_cannot_apply_upper_band', result.reasons)

    def test_area_filters_preserve_rejected_provenance(self):
        arrays = color, mask, depth, _ = scene()
        mask[50, 50] = True
        color[50, 50] = (35, 35, 210)
        depth[50, 50] = 970
        result = run(arrays)
        self.assertEqual(result.count, 1)
        self.assertTrue(result.rejected_mask[50, 50])
        self.assertTrue(result.uncertain_mask[50, 50])

    def test_bgr_noncontiguous_float_meters_nonmutation_and_snapshots(self):
        color, mask, depth, empty = arrays = scene()
        originals = [a.copy() for a in arrays]
        expected = run(arrays)
        actual = run((color[:, ::-1, ::-1], mask[:, ::-1],
                      depth[:, ::-1].astype(float)*.001, empty[:, ::-1].astype(float)*.001),
                     color_order='BGR', depth_scale_m=1., diagnostics=True)
        self.assertEqual(actual.count, expected.count)
        np.testing.assert_array_equal(actual.object_mask[:, ::-1], expected.object_mask)
        for a, b in zip(arrays, originals):
            np.testing.assert_array_equal(a, b)
        mask[:] = False
        depth[:] = 0
        color[:] = 0
        self.assertEqual(actual.candidate_mask.sum(), 400)
        self.assertTrue(actual.observed_mask.all())
        self.assertTrue((actual.diagnostics['depth_difference_m'][actual.object_mask] > 0).all())

    def test_mask_invariants_json_debug_and_determinism(self):
        arrays = scene()
        result = run(arrays, diagnostics=True)
        self.assertEqual(result.to_dict(), run(arrays).to_dict())
        np.testing.assert_array_equal(result.object_mask, result.label_map > 0)
        self.assertFalse((result.object_mask & ~result.candidate_mask).any())
        self.assertFalse((result.object_mask & result.excluded_mask).any())
        self.assertFalse((result.object_mask & result.rejected_mask).any())
        self.assertEqual(result.render_debug(arrays[0]).dtype, np.uint8)
        json.dumps(result.to_dict(), allow_nan=False)

    def test_input_validation(self):
        arrays = color, mask, depth, empty = scene()
        invalid = [dict(color_order='rgb'), dict(diagnostics=1), dict(registered_to_depth=False),
                   dict(registered_to_depth=1), dict(depth_scale_m=None), dict(depth_scale_m=True),
                   dict(depth_scale_m=0), dict(depth_scale_m=np.nan), dict(depth_scale_m=np.inf),
                   dict(depth=None), dict(background_depth=None), dict(depth=depth.astype(bool)),
                   dict(depth=np.zeros((2, 2))), dict(background_depth=empty.astype(complex)),
                   dict(invalid_values=(np.inf,)), dict(invalid_values=1)]
        for key in ('arm_mask', 'exclusion_mask', 'work_roi', 'color_valid_mask', 'depth_valid_mask',
                    'background_depth_valid_mask', 'observed_mask', 'uncertain_mask', 'cleanup_changed_mask'):
            invalid.extend(({key: np.zeros(mask.shape, np.uint8)}, {key: np.zeros((2, 2), bool)}))
        for options in invalid:
            with self.subTest(keys=list(options)):
                with self.assertRaises(ValueError):
                    run(arrays, **options)
        for bad_mask in (mask.astype(np.uint8), mask[:2], mask[..., None]):
            with self.assertRaises(ValueError):
                run((color, bad_mask, depth, empty))
        for bad_color in (color.astype(float), color[..., 0], color[..., :2], color[:1]):
            with self.assertRaises(ValueError):
                run((bad_color, mask, depth, empty))

    def test_mask_only_rejects_unusable_depth_options(self):
        color, mask, _, _ = scene()
        for options in (dict(depth_scale_m=.001), dict(depth_valid_mask=mask),
                        dict(background_depth_valid_mask=mask), dict(invalid_values=(65535,))):
            with self.assertRaises(ValueError):
                OverheadObjectCounter().detect_mask(color, mask, color_order='RGB', **options)

    def test_noise_example_exposes_losses_and_wrong_identity(self):
        process = subprocess.run([sys.executable, 'examples/example_object_count_noise.py'],
                                 check=True, capture_output=True, text=True)
        examples = json.loads(process.stdout)
        self.assertEqual(examples['arm_edge_guard_2']['count'], 2)
        self.assertEqual(examples['arm_edge_guard_2']['pixels']['arm']['grouped'], 236)
        self.assertEqual(examples['arm_edge_guard_3']['count'], 1)
        self.assertEqual(examples['adjacent_strip_guard_0']['count'], 1)
        self.assertEqual(examples['adjacent_strip_guard_2']['count'], 0)
        self.assertEqual(examples['adjacent_strip_guard_2']['pixels']['target']['grouped'], 0)
        self.assertEqual(examples['eight_fragments_min_8']['count'], 2)
        self.assertEqual(examples['eight_fragments_min_10']['count'], 1)
        swapped = examples['upper_band_wrong_identity']
        self.assertEqual(swapped['count'], 1)
        self.assertEqual(swapped['pixels']['target']['grouped'], 0)
        self.assertEqual(swapped['pixels']['arm']['grouped'], 1600)
        contaminated = examples['same_color_contamination']
        self.assertEqual(contaminated['count'], 1)
        self.assertEqual(contaminated['pixels']['target']['grouped'], 400)
        self.assertEqual(contaminated['pixels']['residue']['grouped'], 400)

    def test_example_is_strict_json(self):
        process = subprocess.run([sys.executable, 'examples/example_object_count_mask.py'],
                                 check=True, capture_output=True, text=True)
        examples = json.loads(process.stdout)
        self.assertEqual(examples['external_mask_with_depth']['count'], 2)
        self.assertIsNone(examples['mask_only']['count'])
        self.assertEqual(examples['mask_only']['candidate_count'], 2)
        self.assertIsNone(examples['cleaned_mask_with_provenance']['count'])
        self.assertEqual(examples['cleaned_mask_with_provenance']['metrics']['cleanup_changed_pixels'], 4)


if __name__ == '__main__':
    unittest.main()
