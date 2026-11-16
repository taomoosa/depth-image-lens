"""Opt-in color-core training guard; never learns from inferred foreground.

The support region includes same-color pixels outside the depth seed, so a thin
band of erroneous depth cannot validate itself by selecting only positive votes.
Crop/ROI boundaries are unobserved, not evidence of a physical color boundary.
"""
import numpy as np
from .debug import _opencv


def select_color_core_training(proposals, lab, training, observed, valid, upper,
                               delta, depth_fg, background_distance, scales, p, diagnostics=False):
    cv = _opencv()
    used = np.zeros(training.shape, bool)
    interiors = used.copy()
    accepted = []
    accepted_sources = used.copy()
    accepted_indices = set()
    fit_masks = {}
    records = []
    failure_masks = {}
    radius = p['model_training_inset_px']
    tolerance = p['model_color_tolerance'] * .5
    for index, color in enumerate(proposals):
        distance = np.linalg.norm((lab - color) * scales, axis=2)
        same_color = observed & (distance <= tolerance)
        _, labels = cv.connectedComponents(same_color.astype(np.uint8), connectivity=8)
        ids = np.unique(labels[training & same_color]); ids = ids[ids != 0]
        for component in ids:
            group = labels == component
            # Erode relative to observed different-color pixels only. Invalid
            # depth is NOT an absent color observation and remains in coverage.
            if radius:
                other = observed & ~group
                near_other = cv.dilate(other.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)).astype(bool)
                interior = group & ~near_other
            else:
                interior = group
            interiors |= interior
            sample = interior & valid & ~upper
            area = int(interior.sum()); count = int(sample.sum())
            coverage = count / area if area else 0.
            fraction = float(depth_fg[sample].mean()) if count else 0.
            mean = float(np.clip(delta[sample], -p['local_depth_clip_m'], p['local_depth_clip_m']).mean()) if count else 0.
            candidates = training & interior
            background_ok = background_distance > max(0., p['model_background_margin'])
            eligible = candidates & background_ok
            failures = []
            if area == 0: failures.append('no_color_interior')
            if coverage < p['local_min_valid_fraction']: failures.append('insufficient_valid_depth')
            if fraction < p['local_min_depth_fraction']: failures.append('insufficient_same_color_depth_fraction')
            if mean < p['local_mean_depth_m']: failures.append('insufficient_same_color_mean_depth')
            if int(eligible.sum()) < p['model_min_training_pixels']: failures.append('insufficient_interior_training_pixels')
            record = dict(proposal_index=index, color_lab=color.tolist(), color_area=int(group.sum()),
                          interior_area=area, valid_fraction=coverage, depth_fraction=fraction,
                          mean_depth_m=mean, raw_training_pixels=int((training & group).sum()),
                          eligible_training_pixels=int(eligible.sum()),
                          background_excluded_pixels=int((candidates & ~background_ok).sum()),
                          failures=failures, accepted=False)
            if not failures:
                if index in accepted_indices or len(accepted) < p['model_max_colors']:
                    # Fit only to verified interior evidence. The initial
                    # prototype is a proposal, never the final learned color.
                    if index not in accepted_indices:
                        accepted.append(color); accepted_indices.add(index)
                        fit_masks[index] = np.zeros(training.shape, bool)
                    fit_masks[index] |= eligible
                    used |= eligible; accepted_sources |= group; record['accepted'] = True
                else:
                    record['failures'] = ['color_budget_exceeded']
            if diagnostics:
                for reason in record["failures"]:
                    key = "model_training_" + reason + "_mask"
                    if key not in failure_masks: failure_masks[key] = np.zeros(training.shape, bool)
                    failure_masks[key] |= group
            records.append(record)
    accepted = [np.median(lab[mask], axis=0) for mask in fit_masks.values()]
    return accepted, used, interiors, records, accepted_sources, failure_masks
