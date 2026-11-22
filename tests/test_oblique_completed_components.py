"""First-class mask outputs retain measured identity and completion provenance."""
import json
import unittest

import numpy as np

from depth_image_lens import ObjectOnlyGraspDetector, ObliqueComponentCompletion, TopEntryBridgeDetector
from .test_oblique_invalid_depth_gaps import fixture


def detectors(gap=3):
    return (ObjectOnlyGraspDetector(min_width_px=60, invalid_gap_max_px=gap),
            TopEntryBridgeDetector(gap_radius_px=0, invalid_gap_max_px=gap))


def run(z, delta, cloth, arms, *, gap=3, diagnostics=False, **common):
    a, b = detectors(gap)
    valid = np.isfinite(z) & (z > 0)
    return (
        a.detect(z, cloth & valid, depth_unit='m', difference=delta, diagnostics=diagnostics, **common),
        b.detect_mask(z, (cloth | arms) & valid, depth_unit='m', diagnostics=diagnostics, **common),
    )


class ObliqueCompletionTests(unittest.TestCase):
    def test_completed_pair_masks_without_diagnostics(self):
        z, delta, cloth, arms = fixture()
        z[:,46:49] = 0.
        a, b = run(z, delta, cloth, arms)
        with_debug = run(z, delta, cloth, arms, diagnostics=True)
        baseline = run(z, delta, cloth, arms, gap=0)
        self.assertEqual((a.held, a.bilateral, b.bridge), ('held','unknown','unknown'))
        self.assertEqual(a.metrics['visible_width_px'],30)
        for i, (r, debug, old) in enumerate(zip((a,b), with_debug, baseline)):
            with self.subTest(detector=type(r).__name__):
                self.assertEqual(r.diagnostics, {})
                self.assertIsInstance(r.completion, ObliqueComponentCompletion)
                self.assertEqual(r.completion.status, 'merged')
                owner = r.completion.merged_component_id
                self.assertEqual(owner, 1)
                self.assertEqual(len(r.completed_components),1)
                record, = r.completed_components
                self.assertEqual(record['source_component_ids'],[1,2])
                self.assertEqual(record['inferred_pixels'],24)
                self.assertEqual(record['pixels'],int(r.completed_mask.sum()))
                self.assertEqual(record['measured_pixels'],int(r.completion.measured_mask.sum()))
                self.assertEqual(record['span_x_px'],62)
                np.testing.assert_array_equal(r.completed_mask,r.completed_labels > 0)
                np.testing.assert_array_equal(r.completed_mask,r.completion.measured_mask | r.inferred_mask)
                np.testing.assert_array_equal(r.completed_mask, r.completion.component_mask(owner))
                np.testing.assert_array_equal(r.inferred_mask,debug.diagnostics['invalid_depth_gap_inferred_mask'])
                self.assertFalse((r.inferred_mask & (z > 0)).any())
                self.assertEqual(old.metrics,{k:v for k,v in r.metrics.items() if k != 'invalid_depth_gap'})
                if i == 0:
                    np.testing.assert_array_equal(r.completion.measured_labels, debug.diagnostics['component_labels'])
                else:
                    measured = debug.diagnostics['search_mask'] & (cloth | arms) & (z > 0)
                    np.testing.assert_array_equal(r.completion.measured_mask, measured)
                exported = r.to_dict()
                self.assertNotIn('completed_mask',exported['completion'])
                self.assertNotIn('measured_labels',exported['completion'])
                json.dumps(exported, allow_nan=False)
                exported['completion']['completed_components'][0]['source_component_ids'].clear()
                self.assertEqual(record['source_component_ids'],[1,2])
                for invalid in (True,0,2,1.,'1'):
                    with self.assertRaises(ValueError):
                        r.completion.component_mask(invalid)

    def test_unrelated_measured_component_touching_filled_gap_keeps_its_id(self):
        z, delta, cloth, arms = fixture()
        z[:,46:49] = 0.
        third = np.zeros_like(cloth)
        third[30:34,46:49] = True
        # The island touches the new binary extension in 2D, but lies on a
        # different measured depth surface and must not acquire the merged ID.
        cloth |= third
        z[third] = .65
        delta[third] = .05
        for r in run(z, delta, cloth, arms):
            with self.subTest(detector=type(r).__name__):
                self.assertEqual(r.completion.status,'merged')
                self.assertTrue(r.inferred_mask[29,47])
                self.assertTrue(r.completed_mask[30,47])
                labels = r.completion.measured_labels
                third_id, = np.unique(labels[third])
                other = r.completion.component_mask(int(third_id))
                np.testing.assert_array_equal(other,third)
                np.testing.assert_array_equal(r.completed_labels[third],labels[third])
                root = r.completion.merged_component_id
                self.assertNotEqual(root,third_id)
                self.assertFalse(r.completion.component_mask(root)[third].any())
                record, = [c for c in r.completed_components if c['id']==third_id]
                self.assertEqual(record['source_component_ids'],[third_id])
                self.assertEqual(record['inferred_pixels'],0)
                np.testing.assert_array_equal(r.completed_mask,r.completion.measured_mask | r.inferred_mask)

    def test_opt_in_no_candidate_returns_original_measured_components(self):
        z, delta, cloth, arms = fixture()
        cloth[:,46:49] = False
        z[:,46:49] = 1.5  # Actually measured background, not dropout.
        for r in run(z,delta,cloth,arms):
            self.assertEqual(r.completion.status,'measured_only')
            self.assertIsNone(r.completion.merged_component_id)
            self.assertFalse(r.inferred_mask.any())
            self.assertEqual(len(r.completed_components),2)
            np.testing.assert_array_equal(r.completed_mask,r.completion.measured_mask)
            np.testing.assert_array_equal(r.completed_labels,r.completion.measured_labels)
            self.assertFalse(np.shares_memory(r.completed_labels,r.completion.measured_labels))
            for c in r.completed_components:
                self.assertEqual(c['source_component_ids'],[c['id']])
                self.assertEqual(c['measured_pixels'],c['pixels'])
                self.assertTrue(r.completion.component_mask(c['id']).any())

    def test_unavailable_and_disabled_outputs_are_not_fabricated(self):
        z, delta, cloth, arms = fixture()
        for r in run(z,delta,cloth,arms,gap=0):
            self.assertIsNone(r.completion)
            self.assertIsNone(r.completed_mask)
            self.assertIsNone(r.completed_components)
            self.assertNotIn('completion',r.to_dict())
        a,b=detectors()
        no_entries=b.detect_mask(z,cloth,depth_unit='m')
        self.assertIsNone(no_entries.completion)
        self.assertIsNone(no_entries.completed_labels)
        missing_reference=a.detect(z,cloth,depth_unit='m')
        self.assertEqual(missing_reference.completion.status,'measured_only')
        self.assertEqual(missing_reference.completed_components,[])
        self.assertFalse(missing_reference.completed_mask.any())

    def test_top_completion_does_not_restore_rejected_input(self):
        z, delta, cloth, arms = fixture()
        z[:,46:49] = 0.
        z[45:48,:5] = .95
        foreground=(cloth | arms)
        foreground[45:48,:5] = True  # A valid foreground item outside the search.
        foreground[29,43] = True
        exclude=np.zeros(z.shape,bool);exclude[29,43]=True
        _,b=detectors()
        r = b.detect_mask(z,foreground,depth_unit='m',exclude_mask=exclude,diagnostics=True)
        self.assertEqual(r.completion.status,'merged')
        self.assertFalse(r.completed_mask[:,0:5].any())
        self.assertFalse(r.completed_mask[0].any())  # Not the entire raw entry silhouette.
        self.assertFalse(r.completed_mask[exclude].any())
        self.assertFalse(r.completed_mask[~r.diagnostics['search_mask']].any())


if __name__ == '__main__':
    unittest.main()
