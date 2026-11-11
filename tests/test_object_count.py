"""Synthetic evidence-contract checks; not field accuracy measurements."""
import json
import subprocess
import sys
import unittest
import numpy as np
from depth_image_lens import OverheadObjectCounter, ObjectCountResult, RGBDBackgroundSegmenter


def scene(shape=(80, 100)):
    background_color = np.full((*shape, 3), 150, np.uint8)
    background_depth = np.full(shape, 1000, np.uint16)
    return background_color.copy(), background_depth.copy(), background_color, background_depth


def paint(c, z, box, color=(210, 35, 35), depth=980):
    c[box] = color
    z[box] = depth


def detect(arrays, detector=None, **kwargs):
    c, z, bc, bz = arrays
    return (detector or OverheadObjectCounter()).detect(
        c, z, background_color=bc, background_depth=bz, color_order='RGB',
        registered_to_depth=True, depth_scale_m=.001, **kwargs)


class ObjectCountTests(unittest.TestCase):
    def test_observed_empty_is_zero_not_forced_cluster(self):
        r = detect(scene())
        self.assertIsInstance(r, ObjectCountResult)
        self.assertEqual((r.count, r.candidate_count, r.state), (0, 0, 'no_visible_candidates'))
        self.assertFalse(r.label_map.any())
        self.assertTrue(r.observed_mask.all())

    def test_two_separated_colors_and_group_mask_copy(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        paint(c, z, np.s_[40:65, 55:85], (35, 35, 210))
        r = detect(arrays)
        self.assertEqual(r.count, 2)
        self.assertEqual(sorted(g['area_px'] for g in r.groups), [625, 750])
        mask = r.group_mask(1)
        mask[:] = False
        self.assertTrue(r.group_mask(1).any())
        for value in (0, 3, True, np.int64(1)):
            with self.assertRaises(ValueError):
                r.group_mask(value)

    def test_touching_equal_height_colors_are_two(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        paint(c, z, np.s_[10:35, 35:60], (35, 35, 210))
        r = detect(arrays)
        self.assertEqual(r.count, 2)
        self.assertTrue(r.object_mask[10:35, 10:60].all())

    def test_overlapping_objects_and_disconnected_fragments(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[20:60, 10:85])
        paint(c, z, np.s_[10:70, 35:60], (35, 35, 210), 960)
        r = detect(arrays)
        self.assertEqual(r.count, 2)
        self.assertEqual(sorted(g['fragment_count'] for g in r.groups), [1, 2])

    def test_fully_hidden_color_does_not_add_an_object(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[20:50, 20:50])
        paint(c, z, np.s_[10:60, 10:60], (35, 35, 210), 960)
        self.assertEqual(detect(arrays).count, 1)

    def test_half_shadow_and_smooth_brightness_ramp_stay_one(self):
        for ramp in (False, True):
            arrays = c, z, _, _ = scene()
            paint(c, z, np.s_[10:55, 10:80])
            scale = np.linspace(.3, 1., 70) if ramp else np.r_[np.full(35, .4), np.ones(35)]
            c[10:55, 10:80] = np.rint(np.array([210, 35, 35])[None, None, :] * scale[None, :, None])
            r = detect(arrays)
            self.assertEqual(r.count, 1, r.to_dict())

    def test_small_chromatic_variation_and_red_hue_wrap(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35], (220, 25, 35))
        paint(c, z, np.s_[10:35, 35:60], (220, 35, 25))
        self.assertEqual(detect(arrays).count, 1)

    def test_small_noise_across_histogram_bins_preserves_identity(self):
        rng = np.random.default_rng(23)
        for color in ((210, 35, 35), (35, 210, 35), (35, 35, 210), (120, 80, 40)):
            arrays = c, z, _, _ = scene()
            paint(c, z, np.s_[10:60, 10:80], color)
            noise = rng.integers(-3, 4, (50, 70, 3))
            c[10:60, 10:80] = np.clip(np.asarray(color) + noise, 1, 255).astype(np.uint8)
            r = detect(arrays)
            self.assertEqual(r.count, 1, r.to_dict())

    def test_registration_declaration_is_not_an_alignment_verification(self):
        arrays = c, z, _, _ = scene()
        c[10:35, 10:35] = (210, 35, 35)
        z[10:35, 40:65] = 980  # Deliberately shifted modalities.
        r = detect(arrays)
        self.assertIsNone(r.count)
        self.assertIn('insufficient_depth_support_shadow_or_thin_object', r.reasons)
        self.assertEqual(r.parameters['registration'], 'caller_declared')

    def test_crop_and_nonrectangular_roi_keep_visible_identity(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[:30, :30])
        r = detect(arrays)
        self.assertEqual(r.count, 1)
        self.assertTrue(r.groups[0]['touches_crop'])
        roi = np.ones(z.shape, bool)
        roi[:, 15:] = False
        r = detect(arrays, work_roi=roi)
        self.assertEqual(r.count, 1)
        self.assertTrue(r.groups[0]['touches_crop'])
        self.assertFalse(r.candidate_mask[~roi].any())

    def test_small_fragments_collectively_support_one_identity(self):
        arrays = c, z, _, _ = scene()
        for y in (10, 20, 30):
            for x in (10, 20, 30):
                paint(c, z, np.s_[y:y+3, x:x+3])
        r = detect(arrays)
        self.assertEqual(r.count, 1)
        self.assertEqual(r.groups[0]['fragment_count'], 9)
        self.assertEqual(r.groups[0]['area_px'], 81)

    def test_tiny_distinct_color_beside_dominant_object_is_not_sampled_away(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[5:70, 5:70])
        paint(c, z, np.s_[30:38, 80:88], (35, 35, 210))
        self.assertEqual(detect(arrays).count, 2)

    def test_single_pixel_noise_is_filtered_with_provenance(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        c[60, 60] = (35, 35, 210)
        z[60, 60] = 970
        r = detect(arrays)
        self.assertEqual(r.count, 1)
        self.assertTrue(r.rejected_mask[60, 60])
        self.assertTrue(r.uncertain_mask[60, 60])
        self.assertTrue(any('filtered' in warning for warning in r.warnings))

    def test_arm_mask_guard_suppresses_imperfect_edge(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        paint(c, z, np.s_[10:60, 60:80], (200, 140, 100), 700)
        arm = np.zeros(z.shape, bool)
        arm[11:59, 61:79] = True
        r = detect(arrays, arm_mask=arm)
        self.assertEqual(r.count, 1)
        self.assertFalse(r.object_mask[10:60, 60:80].any())
        self.assertTrue(r.excluded_mask[10:60, 60:80].all())
        self.assertFalse(r.observed_mask[arm].any())

    def test_explicit_upper_camera_z_band_filters_high_arm(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        paint(c, z, np.s_[10:60, 60:80], (200, 140, 100), 700)
        r = detect(arrays, OverheadObjectCounter(max_reference_separation_m=.15))
        self.assertEqual(r.count, 1)
        self.assertEqual(r.metrics['high_separation_pixels'], 1000)
        # No universal depth band is silently applied by default.
        self.assertEqual(detect(arrays).candidate_count, 2)

    def test_missing_depth_cannot_resurrect_upper_band_occluder(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35], (200, 140, 100), 0)
        r = detect(arrays, OverheadObjectCounter(max_reference_separation_m=.15))
        self.assertIsNone(r.count)
        self.assertEqual(r.candidate_count, 0)
        self.assertIn('missing_depth_cannot_apply_upper_band', r.reasons)
        self.assertFalse(r.object_mask.any())

    def test_near_table_arm_is_known_nonsemantic_counterexample(self):
        arrays = c, z, _, _ = scene()
        # This could equally be cloth or an arm. No skin-color model is claimed.
        paint(c, z, np.s_[10:35, 10:35], (200, 140, 100), 980)
        r = detect(arrays, OverheadObjectCounter(max_reference_separation_m=.15))
        self.assertEqual(r.candidate_count, 1)
        self.assertIn('arm_masks_depth_bands_and_area_filters_are_not_semantic_arm_detection', r.warnings)

    def test_color_only_thin_object_and_colored_shadow_abstain(self):
        for current_depth in (1000, 999, 0):
            arrays = c, z, _, _ = scene()
            paint(c, z, np.s_[10:35, 10:35], depth=current_depth)
            r = detect(arrays)
            self.assertEqual(r.candidate_count, 1)
            self.assertIsNone(r.count)
            self.assertIn('insufficient_depth_support_shadow_or_thin_object', r.reasons)

    def test_shadow_cannot_borrow_support_from_distant_same_color(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        paint(c, z, np.s_[40:65, 60:85], depth=1000)
        r = detect(arrays)
        self.assertEqual(r.candidate_count, 1)
        self.assertIsNone(r.count)
        self.assertIn('fragment_without_depth_support', r.reasons)
        self.assertEqual(r.groups[0]['unsupported_fragment_pixels'], 625)

    def test_dark_table_shadow_is_not_certified_as_object(self):
        arrays = c, z, _, _ = scene()
        c[10:35, 10:35] = 35
        r = detect(arrays)
        self.assertIsNone(r.count)
        self.assertIn('achromatic_brightness_identity_ambiguous', r.reasons)
        c[10:35, 10:35] = 120  # Below background-change tolerance.
        self.assertEqual(detect(arrays).count, 0)

    def test_achromatic_objects_are_hypotheses_not_separable_identities(self):
        for rgb in ((30, 30, 30), (150, 150, 150), (255, 255, 255)):
            arrays = c, z, _, _ = scene()
            paint(c, z, np.s_[10:35, 10:35], rgb)
            r = detect(arrays)
            self.assertIsNone(r.count)
            self.assertEqual(r.candidate_count, 1)
            self.assertIn('achromatic_brightness_identity_ambiguous', r.reasons)

    def test_near_black_foreground_not_arbitrary_color_group(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35], (10, 2, 2))
        r = detect(arrays)
        self.assertIsNone(r.count)
        self.assertEqual(r.candidate_count, 0)
        self.assertIn('foreground_too_dark_for_color_identity', r.reasons)

    def test_brightness_only_objects_are_documented_nonidentifiability(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35], (120, 80, 40))
        paint(c, z, np.s_[10:35, 35:60], (240, 160, 80))
        r = detect(arrays)
        self.assertEqual(r.candidate_count, 1)  # Equally consistent with one shaded object.
        self.assertIn('brightness_only_and_arbitrarily_close_colors_cannot_be_reliably_separated', r.warnings)
        self.assertEqual(r.scope, 'visible_distinguishable_color_groups_not_verified_physical_objects')

    def test_color_budget_overflow_is_not_exact_capped_count(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        paint(c, z, np.s_[40:65, 60:85], (35, 35, 210))
        r = detect(arrays, OverheadObjectCounter(max_color_groups=1))
        self.assertIsNone(r.count)
        self.assertIn('color_group_budget_exceeded', r.reasons)
        self.assertTrue(r.rejected_mask.any())

    def test_gradient_does_not_single_link_merge_endpoints(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:60, 10:90])
        colors = np.stack([np.linspace(210, 35, 80), np.full(80, 35), np.linspace(35, 210, 80)], axis=1)
        c[10:60, 10:90] = np.rint(colors).astype(np.uint8)
        r = detect(arrays)
        self.assertGreater(r.candidate_count, 1)
        self.assertIsNone(r.count)
        self.assertTrue(any(reason in r.reasons for reason in ('nearby_color_groups_may_be_shading', 'count_changes_with_color_tolerance', 'broad_color_distribution')))

    def test_invalid_observation_empty_roi_and_exclusion_not_known_empty(self):
        arrays = c, z, _, _ = scene()
        z[:] = 0
        r = detect(arrays)
        self.assertEqual(r.state, 'insufficient')
        self.assertIsNone(r.count)
        self.assertEqual(r.candidate_count, 0)
        for keyword in ('work_roi', 'exclusion_mask'):
            mask = np.zeros(z.shape, bool) if keyword == 'work_roi' else np.ones(z.shape, bool)
            r = detect(scene(), **{keyword: mask})
            self.assertIsNone(r.count)
            self.assertIn('no_searchable_pixels', r.reasons)

    def test_missing_any_modality_is_preserved(self):
        for key in ('color_valid_mask', 'background_color_valid_mask', 'depth_valid_mask', 'background_depth_valid_mask'):
            arrays = c, z, _, _ = scene()
            paint(c, z, np.s_[10:35, 10:35])
            mask = np.ones(z.shape, bool)
            mask[60:65, 60:65] = False
            r = detect(arrays, **{key: mask})
            self.assertIsNone(r.count)
            self.assertFalse(r.observed_mask[60:65, 60:65].any())
            self.assertTrue(r.uncertain_mask[60:65, 60:65].all())

    def test_farther_conflict_sentinel_and_nonfinite_not_unsigned_foreground(self):
        for bad in (1100., 0., -1., np.nan, np.inf, 65535.):
            c, z, bc, bz = scene()
            z = z.astype(float)
            paint(c, z, np.s_[10:35, 10:35], depth=bad)
            r = detect((c, z, bc, bz), invalid_values=(65535.,))
            self.assertIsNone(r.count)
            if bad == 1100:
                self.assertFalse(r.object_mask.any())
                self.assertIn('color_depth_conflict', r.reasons)
            json.dumps(r.to_dict(), allow_nan=False)

    def test_rgb_bgr_float_meters_noncontiguous_and_no_mutation(self):
        arrays = c, z, bc, bz = scene()
        paint(c, z, np.s_[10:35, 10:35])
        original = [a.copy() for a in arrays]
        expected = detect(arrays)
        actual = OverheadObjectCounter().detect(c[:, ::-1, ::-1], z[:, ::-1].astype(float)*.001,
            background_color=bc[:, ::-1, ::-1], background_depth=bz[:, ::-1].astype(float)*.001,
            color_order='BGR', registered_to_depth=True, depth_scale_m=1.)
        self.assertEqual(actual.count, expected.count)
        np.testing.assert_array_equal(actual.object_mask[:, ::-1], expected.object_mask)
        for a, b in zip(arrays, original):
            np.testing.assert_array_equal(a, b)

    def test_determinism_mask_invariants_and_debug(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        r = detect(arrays, diagnostics=True)
        self.assertEqual(r.to_dict(), detect(arrays).to_dict())
        np.testing.assert_array_equal(r.object_mask, r.label_map > 0)
        self.assertFalse((r.object_mask & ~r.candidate_mask).any())
        self.assertFalse((r.object_mask & r.excluded_mask).any())
        self.assertFalse((r.rejected_mask & r.object_mask).any())
        self.assertEqual(r.render_debug(c).dtype, np.uint8)
        json.dumps(r.to_dict(), allow_nan=False)

    def test_custom_segmenter_and_rejected_reference(self):
        arrays = c, z, _, _ = scene()
        c[:] = (10, 220, 40)
        check = np.zeros(z.shape, bool)
        check[:10] = True
        r = detect(arrays, background_check_mask=check)
        self.assertIsNone(r.count)
        self.assertIn('background_color_reference_unusable', r.reasons)
        r = detect(scene(), OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(fusion_mode='depth_required')))
        self.assertEqual(r.count, 0)

    def test_custom_fusion_suppression_and_filtering_are_not_empty(self):
        for mode in ('evidence', 'color_seeded'):
            arrays = c, z, _, _ = scene()
            z[10:35, 10:35] = 993  # Unconfirmed same-background-color change.
            r = detect(arrays, OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(fusion_mode=mode)))
            self.assertIsNone(r.count)
            self.assertEqual(r.candidate_count, 0)
            self.assertIn('background_fusion_has_unresolved_changes', r.reasons)
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35])
        r = detect(arrays, OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(min_object_area=1000)))
        self.assertIsNone(r.count)
        self.assertIn('background_segmentation_filtered_substantial_change', r.reasons)

    def test_input_contracts(self):
        c, z, bc, bz = scene()
        base = dict(background_color=bc, background_depth=bz, color_order='RGB',
                    registered_to_depth=True, depth_scale_m=.001)
        bad = [dict(registered_to_depth=False), dict(color_order='rgb'), dict(depth_scale_m=True),
               dict(depth_scale_m=0), dict(diagnostics=1), dict(arm_mask=np.zeros(z.shape, np.uint8)),
               dict(exclusion_mask=np.zeros((2, 2), bool)), dict(background_color=bc.astype(float))]
        for options in bad:
            with self.subTest(options=list(options)):
                with self.assertRaises(ValueError):
                    OverheadObjectCounter().detect(c, z, **dict(base, **options))

    def test_small_tolerance_does_not_premerge_histogram_colors(self):
        arrays = c, z, _, _ = scene()
        paint(c, z, np.s_[10:35, 10:35], (120, 80, 40))
        paint(c, z, np.s_[40:65, 60:85], (121, 80, 40))
        for tolerance in (.01, 1e-300):
            r = detect(arrays, OverheadObjectCounter(color_tolerance=tolerance, color_margin=0.))
            self.assertEqual(r.candidate_count, 2, r.to_dict())
            self.assertEqual(r.parameters['histogram_bin_width'], 0.)
            json.dumps(r.to_dict(), allow_nan=False)

    def test_parameter_contracts(self):
        bad = [dict(color_tolerance=0), dict(color_tolerance=np.nan), dict(color_tolerance=True),
               dict(color_margin=18), dict(min_object_area=0), dict(min_fragment_area=65),
               dict(min_depth_support_fraction=0), dict(min_brightness=0), dict(min_saturation=2),
               dict(exclusion_guard_px=65), dict(max_color_groups=129), dict(max_color_groups=True),
               dict(max_reference_separation_m=0), dict(max_reference_separation_m=np.inf),
               dict(stability_fraction=0), dict(background_segmenter={})]
        for options in bad:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    OverheadObjectCounter(**options)

    def test_example_is_strict_json_and_self_contained(self):
        process = subprocess.run([sys.executable, 'examples/example_object_count.py'], check=True,
                                 capture_output=True, text=True)
        examples = json.loads(process.stdout)
        self.assertEqual(examples['empty']['count'], 0)
        self.assertEqual(examples['overlap_shading_crop']['count'], 2)
        self.assertEqual(examples['imperfect_arm_mask']['count'], 1)
        self.assertIsNone(examples['thin_or_shadow']['count'])


if __name__ == '__main__':
    unittest.main()
