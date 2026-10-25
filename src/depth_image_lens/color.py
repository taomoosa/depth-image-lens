"""Seeded color assistance inside depth foreground; no semantic hand model."""
from dataclasses import dataclass, field
import numpy as np

from .bridge_drape import _mask
from .depth import depth_pair_16uc1_to_meters


def _erode(mask, radius):
    if not radius:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    result = mask.copy()
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy:dy + h, dx:dx + w]
    return result


def _distance(colors, samples):
    # Bound palette size and memory. This is a deterministic sample, not training.
    palette = np.unique(samples, axis=0)
    if len(palette) > 64:
        palette = palette[np.linspace(0, len(palette) - 1, 64).astype(int)]
    nearest = np.full(len(colors), np.inf)
    for sample in palette:
        delta = colors - sample.astype(float)
        nearest = np.minimum(nearest, np.einsum('ij,ij->i', delta, delta))
    return np.sqrt(nearest)


@dataclass(frozen=True)
class ColorAssistResult:
    object_mask: np.ndarray = field(repr=False)
    hand_mask: np.ndarray = field(repr=False)
    uncertain_mask: np.ndarray = field(repr=False)
    summary: dict
    hand_exclusion_mask: np.ndarray | None = field(default=None, repr=False)
    appearance_uncertain_mask: np.ndarray | None = field(default=None, repr=False)
    object_appearance_candidate_mask: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self):
        """Strict-JSON summary; masks remain arrays for caller inspection."""
        return dict(self.summary)


class SeededColorAssistant:
    """RGB-distance candidates constrained by depth, with explicit abstention.

    Seeds identify two appearance classes, not skin color or verified contacts.
    max_color_distance and min_color_margin use Euclidean 8-bit RGB distance.
    boundary_guard_px removes a narrow band from both candidate classes.
    All results require visual review; registration is a caller declaration.
    """
    def __init__(self, *, max_color_distance=60., min_color_margin=15., boundary_guard_px=1):
        if not np.isfinite([max_color_distance, min_color_margin]).all() or min(max_color_distance, min_color_margin) <= 0:
            raise ValueError('positive finite color thresholds required')
        if type(boundary_guard_px) is not int or not 0 <= boundary_guard_px <= 3:
            raise ValueError('boundary_guard_px must be an integer from 0 to 3')
        self.max_color_distance = float(max_color_distance)
        self.min_color_margin = float(min_color_margin)
        self.boundary_guard_px = boundary_guard_px

    def segment(self, color, foreground_mask, *, object_seeds, hand_seeds,
                color_order, registered_to_depth, valid_mask=None, color_valid_mask=None):
        """Use aligned uint8 HxWx3 color and bool HxW masks; no input mutation.

        Empty or unusable seed classes cause abstention. Seeds outside supplied
        foreground or validity are ignored and counted, never made valid.
        Similar colors remain uncertain even at seed pixels. Missing pixels are
        not filled. Pixelwise color matching can cross semantic boundaries.
        """
        if registered_to_depth is not True:
            raise ValueError('registered_to_depth=True must be explicitly declared; equal shape is not registration')
        if color_order not in ('RGB', 'BGR'):
            raise ValueError('explicit RGB or BGR color_order required')
        rgb = np.asarray(color)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 2:
            raise ValueError('color must be uint8 HxWx3')
        if color_order == 'BGR':
            rgb = rgb[..., ::-1]
        shape = rgb.shape[:2]
        fg = _mask(foreground_mask, shape, 'foreground_mask')
        os = _mask(object_seeds, shape, 'object_seeds')
        hs = _mask(hand_seeds, shape, 'hand_seeds')
        if np.any(os & hs):
            raise ValueError('seed classes must not overlap')
        valid = np.ones(shape, bool) if valid_mask is None else _mask(valid_mask, shape, 'valid_mask')
        color_valid = np.ones(shape, bool) if color_valid_mask is None else _mask(color_valid_mask, shape, 'color_valid_mask')
        eligible = fg & valid & color_valid
        object_samples = rgb[os & eligible]
        hand_samples = rgb[hs & eligible]
        obj = np.zeros(shape, bool)
        hand = np.zeros(shape, bool)
        reason = 'missing_usable_seed_class'
        if len(object_samples) and len(hand_samples):
            pixels = rgb[eligible].astype(float)
            od = _distance(pixels, object_samples)
            hd = _distance(pixels, hand_samples)
            obj[eligible] = (od <= self.max_color_distance) & (hd - od >= self.min_color_margin)
            hand[eligible] = (hd <= self.max_color_distance) & (od - hd >= self.min_color_margin)
            reason = 'seed_color_distance_candidates' if np.any(obj | hand) else 'appearance_not_separable'
        appearance_candidate=obj.copy()
        if len(object_samples) and not len(hand_samples):
            appearance_candidate[eligible]=_distance(rgb[eligible].astype(float),object_samples)<=self.max_color_distance
        # Keep the full separated hand appearance for exclusion, not for contact.
        # Eroding an arm exclusion can leave an artificial foreground outline path.
        hand_exclusion = hand.copy()
        appearance_uncertain = (fg & ~(obj | hand)) | ~valid
        obj = _erode(obj, self.boundary_guard_px)
        hand = _erode(hand, self.boundary_guard_px)
        uncertain = (fg & ~(obj | hand)) | ~valid
        denominator = int((fg & valid).sum())
        summary = {
            'reason': reason, 'candidate_only': True, 'physical_contact_verified': False,
            'validation_status': 'not_validated', 'registration': 'caller_declared',
            'color_order': color_order, 'foreground_pixels': int(fg.sum()),
            'object_pixels': int(obj.sum()), 'hand_pixels': int(hand.sum()),
            'uncertain_pixels': int(uncertain.sum()),
            'hand_exclusion_pixels': int(hand_exclusion.sum()),
            'appearance_uncertain_pixels': int(appearance_uncertain.sum()),
            'classified_foreground_fraction': float((obj | hand).sum() / denominator) if denominator else None,
            'usable_object_seed_pixels': len(object_samples), 'usable_hand_seed_pixels': len(hand_samples),
            'ignored_seed_pixels': int(((os | hs) & ~eligible).sum()),
            'max_color_distance': self.max_color_distance, 'min_color_margin': self.min_color_margin,
            'boundary_guard_px': self.boundary_guard_px,
            'stage_results': {
                'object_appearance': {'status':'computed' if len(object_samples) else 'insufficient_input',
                    'candidate_pixels':int(appearance_candidate.sum()),'semantic_identity_verified':False},
                'class_separation': {'status':'computed' if len(object_samples) and len(hand_samples) else 'insufficient_input',
                    'reason':reason}},
        }
        return ColorAssistResult(obj, hand, uncertain, summary, hand_exclusion, appearance_uncertain, appearance_candidate)

    def segment_16uc1(self, depth, difference, color, *, object_seeds, hand_seeds,
                      color_order, registered_to_depth, depth_scale_m=.001,
                      threshold_m=.02, observed_mask=None, background_valid=None, color_valid_mask=None):
        """Optional uint16 entry: positive clipped background-current, zero non-foreground.

        Raw zero is invalid. Units are metres/count, never inferred from color.
        The foreground threshold should match the BridgeDrapeDetector instance.
        """
        if not np.isfinite(threshold_m) or threshold_m < 0:
            raise ValueError('threshold_m must be finite and nonnegative')
        z, delta = depth_pair_16uc1_to_meters(depth, difference, depth_scale_m=depth_scale_m)
        valid = z > 0
        for mask, name in ((observed_mask, 'observed_mask'), (background_valid, 'background_valid')):
            if mask is not None:
                valid &= _mask(mask, z.shape, name)
        foreground = valid & (delta > 0) & (delta > threshold_m)
        return self.segment(color, foreground, object_seeds=object_seeds, hand_seeds=hand_seeds,
                            color_order=color_order, registered_to_depth=registered_to_depth,
                            valid_mask=valid, color_valid_mask=color_valid_mask)
