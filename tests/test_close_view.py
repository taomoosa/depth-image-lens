"""Fixed close-view regressions; synthetic derivatives, not real accuracy."""
import json
import unittest

import numpy as np

from depth_image_lens import BridgeDrapeDetector
from .test_bridge_drape import ROIS, scene


def counts(mask):
    return (np.where(mask, 1500, 2000).astype(np.uint16),
            np.where(mask, 500, 0).astype(np.uint16))


def translated(mask, dx):
    result = np.zeros_like(mask)
    if dx >= 0:
        result[:, dx:] = mask[:, :mask.shape[1] - dx]
    else:
        result[:, :dx] = mask[:, -dx:]
    return result


class CloseViewTests(unittest.TestCase):
    def test_lower_two_thirds_translation_and_down(self):
        # Five source shapes reused across translations/directions: not 45 sources.
        detector = BridgeDrapeDetector()
        for kind in ('u', 'long_hem', 'left', 'right', 'separate'):
            for dx in (-7, 0, 9):
                mask = translated(scene(kind), dx)[33:]
                for down in ((0, 1), (.1, 1), (-.1, 1)):
                    with self.subTest(kind=kind, dx=dx, down=down):
                        result = detector.detect_16uc1(
                            *counts(mask), upper_band=(0, 5), down=down)
                        self.assertEqual((result.bridge, result.large_sag, result.drape),
                                         ('unknown', 'unknown', 'unknown'))
                        self.assertEqual(result.reasons['bridge'], 'upper_foreground_clipped')
                        self.assertEqual(result.auto['hand_roi_candidates'], {})
                        self.assertFalse(result.auto['physical_contact_verified'])
                        json.dumps(result.to_dict(), allow_nan=False)

    def test_visible_controls_translation(self):
        detector = BridgeDrapeDetector()
        for kind in ('u', 'left', 'right', 'separate'):
            mask = scene(kind)
            if kind in ('left', 'right'):
                x = 100 if kind == 'left' else 20
                mask[13:21, x - 4:x + 5] = True
            expected_bridge = 'yes' if kind == 'u' else 'no'
            expected_drape = kind if kind in ('left', 'right') else 'both'
            for dx in (-7, 0, 9):
                with self.subTest(kind=kind, dx=dx):
                    result = detector.detect_16uc1(
                        *counts(translated(mask, dx)), upper_band=(8, 20))
                    self.assertEqual((result.bridge, result.drape),
                                     (expected_bridge, expected_drape))

    def test_same_lower_pixels_do_not_reveal_hidden_bridge(self):
        # Two independent drapes and a connected bridge have identical lower crops.
        separate = scene('separate')
        connected = separate | scene('flat')
        self.assertFalse(np.array_equal(separate, connected))
        self.assertTrue(np.array_equal(separate[33:], connected[33:]))
        detector = BridgeDrapeDetector()
        a = detector.detect_16uc1(*counts(separate[33:]), upper_band=(0, 5))
        b = detector.detect_16uc1(*counts(connected[33:]), upper_band=(0, 5))
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(a.bridge, 'unknown')
        # An entirely off-screen bridge leaves an observed empty crop. Its 'no'
        # describes the visible image only, never absence outside the field of view.
        empty = detector.detect_16uc1(*counts(scene('flat')[33:]))
        self.assertEqual(empty.bridge, 'no')
        self.assertEqual(empty.reasons['bridge'], 'no_foreground_in_observed_image')

    def test_manual_hidden_supports_after_crop(self):
        boxes = {key: (x0, y0 - 33, x1, y1 - 33)
                 for key, (x0, y0, x1, y1) in ROIS.items()}
        for kind in ('u', 'left', 'right'):
            result = BridgeDrapeDetector().detect_16uc1(
                *counts(scene(kind)[33:]), boxes, hand_visibility=('hidden', 'hidden'))
            self.assertEqual((result.bridge, result.large_sag, result.drape),
                             ('unknown', 'unknown', 'unknown'))
            self.assertEqual(result.features['anchor_observable'], [False, False])


if __name__ == '__main__':
    unittest.main()
