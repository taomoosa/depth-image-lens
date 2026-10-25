import json
import unittest
import numpy as np
from depth_image_lens import SeededColorAssistant
from .rgbd_cases import cases


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.case = next(cases())
        self.helper = SeededColorAssistant()

    def segment(self, **kwargs):
        c = self.case
        options = dict(object_seeds=c['object_seeds'], hand_seeds=c['hand_seeds'],
                       color_order='RGB', registered_to_depth=True)
        options.update(kwargs)
        return self.helper.segment_16uc1(c['depth'], c['difference'], c['color'], **options)

    def test_disjoint_depth_constrained_candidates(self):
        r = self.segment()
        c = self.case
        self.assertTrue(r.object_mask.any()); self.assertTrue(r.hand_mask.any())
        self.assertFalse(np.any(r.object_mask & r.hand_mask))
        self.assertFalse(np.any(r.object_mask & ~c['object_gt']))
        self.assertFalse(np.any(r.hand_mask & ~c['hand_gt']))
        self.assertFalse(r.summary['physical_contact_verified'])
        json.dumps(r.to_dict(), allow_nan=False)

    def test_similar_colors_abstain_without_overriding_seeds(self):
        self.case = next(c for c in cases() if c['name'] == 'same_color')
        r = self.segment()
        self.assertFalse(r.object_mask.any() or r.hand_mask.any())
        self.assertEqual(r.summary['reason'], 'appearance_not_separable')
        self.assertTrue(r.uncertain_mask[self.case['difference'] > 0].all())

    def test_seed_missing_and_pattern_unknown(self):
        r = self.segment(hand_seeds=np.zeros_like(self.case['hand_seeds']))
        self.assertEqual(r.summary['reason'], 'missing_usable_seed_class')
        self.assertFalse(r.object_mask.any())
        self.case = next(c for c in cases() if c['name'] == 'pattern_unseeded')
        r = self.segment()
        patch = self.case['object_gt'] & (self.case['color'][..., 0] > 200)
        self.assertTrue(r.uncertain_mask[patch].all())

    def test_invalid_depth_zero_difference_and_validity(self):
        c = self.case
        c['depth'][40:60, 58:62] = 0
        valid = np.ones_like(c['object_gt']); valid[:, 90:94] = False
        color_valid = np.ones_like(valid); color_valid[:, 30:34] = False
        r = self.segment(observed_mask=valid, color_valid_mask=color_valid)
        self.assertTrue(r.uncertain_mask[c['depth'] == 0].all())
        self.assertFalse(r.object_mask[~valid].any())
        self.assertFalse(r.object_mask[~color_valid].any())
        self.assertFalse(r.object_mask[c['difference'] == 0].any())
        r = self.segment(threshold_m=0)
        self.assertFalse(r.hand_mask[c['difference'] == 0].any())

    def test_bgr_views_and_no_mutation(self):
        c = self.case
        original = {k:v.copy() for k,v in c.items() if isinstance(v,np.ndarray)}
        rgb = self.segment()
        c['color'] = c['color'][..., ::-1]
        bgr = self.segment(color_order='BGR')
        np.testing.assert_array_equal(rgb.object_mask, bgr.object_mask)
        c['color'] = c['color'][..., ::-1]
        for k,v in original.items(): np.testing.assert_array_equal(c[k],v)

    def test_registration_and_input_contract(self):
        for options in ({'registered_to_depth':False}, {'color_order':'auto'}, {'depth_scale_m':0}):
            with self.assertRaises(ValueError): self.segment(**options)
        with self.assertRaises(ValueError): self.segment(hand_seeds=self.case['object_seeds'])
        self.case['color'] = self.case['color'].astype(float)
        with self.assertRaises(ValueError): self.segment()
        for params in ({'max_color_distance':float('nan')}, {'min_color_margin':0}, {'boundary_guard_px':4}):
            with self.assertRaises(ValueError): SeededColorAssistant(**params)

    def test_metric_scale_equivalence(self):
        a = self.segment()
        self.case['depth'] *= 2; self.case['difference'] *= 2
        b = self.segment(depth_scale_m=.0005)
        np.testing.assert_array_equal(a.object_mask,b.object_mask)



if __name__ == '__main__': unittest.main()
