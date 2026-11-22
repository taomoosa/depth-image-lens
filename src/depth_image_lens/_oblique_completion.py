"""Post-hypothesis component geometry. No inferred depth values."""
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ObliqueComponentCompletion:
    """Aligned, detached geometry and provenance for optional gap completion.

    Label 0 is background. Positive IDs identify original measured components
    or the accepted pair rooted at the smaller original ID. The completed
    labels, not connected-components on the union mask, preserve identity.
    """
    status: str
    scope: str
    merged_component_id: int | None
    completed_components: list
    measured_labels: np.ndarray = field(repr=False)
    completed_labels: np.ndarray = field(repr=False)
    measured_mask: np.ndarray = field(repr=False)
    inferred_mask: np.ndarray = field(repr=False)
    completed_mask: np.ndarray = field(repr=False)

    def component_mask(self, component_id):
        """Return a detached bool mask for one completed component ID."""
        if (isinstance(component_id, (bool, np.bool_))
                or not isinstance(component_id, (int, np.integer))
                or not any(c['id'] == component_id for c in self.completed_components)):
            raise ValueError('component_id must identify an existing completed component')
        return self.completed_labels == component_id

    def to_dict(self):
        """Return JSON-ready metadata without arrays or synthesized depth."""
        return deepcopy(dict(status=self.status, scope=self.scope,
                             merged_component_id=self.merged_component_id,
                             completed_components=self.completed_components))


class _CompletionAccess:
    """Parallel access on detector results; None if completion was not evaluated."""

    @property
    def completed_components(self):
        if self.completion is not None:
            return self.completion.completed_components
        return None

    @property
    def completed_labels(self):
        if self.completion is not None:
            return self.completion.completed_labels
        return None

    @property
    def completed_mask(self):
        if self.completion is not None:
            return self.completion.completed_mask
        return None

    @property
    def inferred_mask(self):
        if self.completion is not None:
            return self.completion.inferred_mask
        return None

    def _completion_metadata(self, base):
        values = {k: v for k, v in base.items() if k not in ('diagnostics', 'completion')}
        if self.completion is not None:
            values['completion'] = self.completion.to_dict()
        return values


def complete_components(labels, *, scope, ids=(), inferred=None):
    """Relabel only the accepted pair; leave all other measured IDs intact."""
    measured_labels = labels.copy()
    measured_mask = measured_labels > 0
    shape = measured_labels.shape
    completed = measured_labels.copy()
    inferred = np.zeros(shape, bool) if inferred is None else inferred.copy()
    source_ids = tuple(sorted(map(int, ids)))
    owner = None
    if source_ids:
        if len(source_ids) != 2 or source_ids[0] >= source_ids[1] or not inferred.any() or (inferred & measured_mask).any():
            raise ValueError('completion requires an accepted pair and unmeasured inferred pixels')
        owner, second = source_ids
        completed[(completed == second) | inferred] = owner

    size = int(measured_labels.max(initial=0)) + 1
    measured_counts = np.bincount(measured_labels.ravel(), minlength=size)
    ypos, xpos = np.nonzero(completed)
    lid = completed[ypos, xpos]
    xmin = np.full(size, shape[1], dtype=np.intp)
    ymin = np.full(size, shape[0], dtype=np.intp)
    xmax = np.full(size, -1, dtype=np.intp)
    ymax = np.full(size, -1, dtype=np.intp)
    np.minimum.at(xmin,lid,xpos); np.minimum.at(ymin,lid,ypos)
    np.maximum.at(xmax,lid,xpos); np.maximum.at(ymax,lid,ypos)
    parts = []
    added = int(inferred.sum())
    for label in np.flatnonzero(xmax >= 0):
        label = int(label)
        parents = source_ids if label == owner else (label,)
        n = sum(int(measured_counts[i]) for i in parents)
        k = added if label == owner else 0
        box = [int(xmin[label]),int(ymin[label]),int(xmax[label])+1,int(ymax[label])+1]
        parts.append(dict(id=label, source_component_ids=list(parents), pixels=n+k,
                          measured_pixels=n, inferred_pixels=k, bbox_xyxy=box,
                          span_x_px=box[2]-box[0], span_y_px=box[3]-box[1]))
    return ObliqueComponentCompletion(
        'merged' if owner is not None else 'measured_only', scope, owner, parts,
        measured_labels, completed, measured_mask, inferred, completed > 0)
