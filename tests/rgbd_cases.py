"""Ten fixed color/depth scenes, not independent real observations."""
import numpy as np
from .test_bridge_drape import scene, ROIS


def cases():
    for name in ('contrast_u', 'arm_false_bridge', 'same_color', 'pattern_seeded',
                 'pattern_unseeded', 'illumination_shift', 'missing_depth',
                 'registration_shift', 'wrong_seed_labels', 'empty'):
        obj = scene('u')
        hand = np.zeros_like(obj)
        hand[10:18, 17:24] = True
        hand[10:18, 97:104] = True
        if name == 'arm_false_bridge':
            obj = scene('left')
            hand |= scene('flat') & ~obj
        if name == 'empty':
            obj[:] = False; hand[:] = False
        fg = obj | hand
        rgb = np.full((*fg.shape, 3), 40, np.uint8)
        rgb[obj] = (30, 70, 200); rgb[hand] = (160, 130, 100)
        os = np.zeros_like(fg); hs = np.zeros_like(fg)
        if obj.any():
            oy, ox = np.argwhere(obj)[len(np.argwhere(obj)) // 3]
            os[oy, ox] = True
        if hand.any():
            hy, hx = np.argwhere(hand)[0]; hs[hy, hx] = True
        patch = obj & (np.indices(fg.shape)[1] >= 55) & (np.indices(fg.shape)[1] <= 65)
        if name == 'same_color': rgb[hand] = (30, 70, 200)
        if name.startswith('pattern_'):
            rgb[patch] = (230, 210, 20)
            if name == 'pattern_seeded':
                y, x = np.argwhere(patch)[len(np.argwhere(patch)) // 2]; os[y, x] = True
        if name == 'illumination_shift': rgb[patch] = (130, 170, 250)
        if name == 'registration_shift': rgb = np.roll(rgb, 8, axis=1)
        if name == 'wrong_seed_labels': os, hs = hs, os
        depth = np.where(fg, 1500, 2000).astype(np.uint16)
        diff = np.where(fg, 500, 0).astype(np.uint16)
        if name == 'missing_depth': depth[:, 59:62] = 0
        yield dict(name=name, depth=depth, difference=diff, color=rgb,
                   object_gt=obj, hand_gt=hand, object_seeds=os, hand_seeds=hs,
                   drape_truth='neither' if name=='empty' else 'left' if name=='arm_false_bridge' else 'both',
                   hand_rois=ROIS, bridge_truth='no' if name in ('arm_false_bridge','empty') else 'yes')


def refinement_negatives():
    """Three explicit derivatives of the same negative; not new observations."""
    for name in ('same_color_arm_negative', 'misaligned_arm_negative', 'unseeded_arm_color_negative'):
        case = next(c for c in cases() if c['name'] == 'arm_false_bridge')
        case['name'] = name
        if name == 'same_color_arm_negative':
            case['color'][case['hand_gt']] = (30, 70, 200)
        elif name == 'misaligned_arm_negative':
            case['color'] = np.roll(case['color'], 8, axis=1)
        else:
            case['color'][case['hand_gt'] & scene('flat')] = (30, 220, 50)
        yield case
