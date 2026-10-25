"""Frozen synthetic conditions; validation geometry does not reuse scene()."""
import numpy as np
from .rgbd_cases import cases, refinement_negatives

CONDITIONS = ('clean', 'anchor_background_1', 'anchor_background_3', 'anchor_object_1',
              'anchor_object_2', 'object_hole_1', 'object_hole_3', 'object_hole_5',
              'object_cut_1', 'object_cut_5', 'anchor_missing', 'remote_holes',
              'search_background_hole', 'unobserved_anchor', 'background_invalid_anchor',
              'excluded_anchor', 'color_invalid_path', 'similar_color', 'pattern_seeded',
              'pattern_unseeded', 'registration_1', 'registration_6', 'left_only',
              'separate', 'robot_false_bridge', 'robot_unseen_color', 'crop', 'empty')


def validation_cases():
    # New 116x156 sloped geometry and colors, fixed before evaluation. Repeated
    # perturbations are correlated conditions of one construction, not 28 captures.
    shape = (116, 156); yy, xx = np.indices(shape)
    for condition in CONDITIONS:
        obj = np.zeros(shape, bool); hands = np.zeros(shape, bool)
        for x in range(27, 129):
            t = (x-27)/101
            top = 26 + round(5*t + 28*np.sin(np.pi*t)**1.5)
            obj[top:top+11, x] = True
        hands[13:24, 22:33] = True; hands[18:29, 123:134] = True
        boxes = {'left': (20, 10, 36, 28), 'right': (120, 15, 136, 33)}
        truth, drape_truth = 'yes', 'both'
        if condition in ('left_only', 'separate', 'robot_false_bridge', 'robot_unseen_color'):
            obj[:] = False; obj[27:83, 25:33] = True
            truth, drape_truth = 'no', 'left'
            if condition == 'separate': obj[32:88, 126:134] = True; drape_truth = 'both'
            if condition.startswith('robot_'):
                # Elbow/bar-shaped support, attached to both anchor candidates.
                for x in range(29, 130):
                    top = 28+round((x-29)*5/100)
                    hands[top:top+10, x] = True
                hands[32:69, 76:86] = True
                hands &= ~obj
        if condition == 'empty':
            obj[:] = False; hands[:] = False; truth, drape_truth = 'no', 'neither'
        fg = obj | hands
        depth = np.where(fg, 1360, 2150).astype(np.uint16)
        difference = np.where(fg, 790, 0).astype(np.uint16)
        color = np.full((*shape, 3), 44, np.uint8)
        color[obj] = (45, 95, 190); color[hands] = (195, 135, 65)
        os = np.zeros(shape, bool); hs = np.zeros(shape, bool)
        if obj.any():
            y, x = np.argwhere(obj)[len(np.argwhere(obj))//4]; os[y, x] = True
        if hands.any(): hs[16, 26] = True
        options = {}
        if condition.startswith('anchor_background_'):
            n = int(condition.rsplit('_', 1)[1]); depth[25:25+n, 20:20+n] = 0
        if condition.startswith('anchor_object_'):
            n = int(condition.rsplit('_', 1)[1]); depth[28:28+n, 29:29+n] = 0
        if condition.startswith('object_hole_'):
            n = int(condition.rsplit('_', 1)[1]); depth[59:59+n, 76:76+n] = 0
        if condition.startswith('object_cut_'):
            n = int(condition.rsplit('_', 1)[1]); depth[:, 77:77+n] = 0
        if condition == 'anchor_missing': depth[24:32, 20:36] = 0
        if condition == 'remote_holes': depth[:8, :15] = 0
        if condition == 'search_background_hole': depth[95:100, 65:70] = 0
        if condition in ('unobserved_anchor', 'background_invalid_anchor'):
            mask = np.ones(shape, bool); mask[25:28, 20:23] = False
            options['observed_mask' if condition == 'unobserved_anchor' else 'background_valid'] = mask
        if condition == 'excluded_anchor':
            mask = np.zeros(shape, bool); mask[24:32, 20:36] = True; options['arm_mask'] = mask
        if condition == 'color_invalid_path':
            mask = np.ones(shape, bool); mask[:, 76:80] = False; options['color_valid_mask'] = mask
        if condition == 'similar_color': color[hands] = (46, 96, 191)
        if condition.startswith('pattern_'):
            patch = obj & (xx >= 73) & (xx < 84); color[patch] = (235, 225, 40)
            if condition == 'pattern_seeded':
                y, x = np.argwhere(patch)[len(np.argwhere(patch))//2]; os[y, x] = True
        if condition.startswith('registration_'): color = np.roll(color, int(condition.rsplit('_', 1)[1]), axis=1)
        if condition == 'robot_unseen_color': color[hands & (yy > 29)] = (30, 225, 75)
        if condition == 'crop': boxes['right'] = (149, 15, 163, 33)
        yield dict(name='validation_'+condition, group='validation', depth=depth, difference=difference,
                   color=color, object_gt=obj, hand_gt=hands, object_seeds=os, hand_seeds=hs,
                   hand_rois=boxes, bridge_truth=truth, drape_truth=drape_truth, options=options,
                   condition=condition, provenance='independent synthetic construction v1; correlated perturbations')


def development_cases():
    for case in list(cases()) + list(refinement_negatives()):
        yield {**case, 'group': 'development', 'options': {}, 'condition': case['name'],
               'provenance': 'original fixed RGB-D regression inputs'}
    # Single-variable causes on the original bridge, separate from validation.
    for label, box in [('anchor_hole', (14, 17, 15, 18)), ('remote_hole', (0, 0, 8, 8)),
                       ('object_pinhole', (60, 58, 61, 59)), ('object_cut_1', (60, 0, 61, 100))]:
        case = next(c for c in cases() if c['name'] == 'contrast_u')
        x0, y0, x1, y1 = box; case['depth'][y0:y1, x0:x1] = 0
        yield {**case, 'name': 'development_'+label, 'group': 'development', 'options': {},
               'condition': label, 'provenance': 'original contrast scene missingness derivative'}


def provenance_challenges():
    """Post-gate input-semantics audit; never used to tune the fixed gate.

    Underlying object truth remains yes. Incorrect zero-difference masking may
    yield a false negative against that truth; report it rather than relabel GT.
    """
    for condition in ('zero_difference_path_cut', 'zero_difference_anchor',
                      'unobserved_path_cut', 'background_invalid_path_cut'):
        c = next(validation_cases())
        if condition == 'zero_difference_path_cut': c['difference'][:, 77:80] = 0
        elif condition == 'zero_difference_anchor': c['difference'][24:32, 20:36] = 0
        else:
            mask = np.ones(c['depth'].shape, bool); mask[:, 77:80] = False
            c['options']['observed_mask' if condition.startswith('unobserved') else 'background_valid'] = mask
        c.update(name='provenance_'+condition, group='input_provenance_challenge', condition=condition,
                 provenance='post-gate input-semantics audit; same validation clean shape, not independent evidence')
        yield c
