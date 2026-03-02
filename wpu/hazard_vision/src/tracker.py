"""Track history: dict[id, deque[TrackedObject]] for TTC/growth computation."""

from collections import deque
from typing import Deque

from .utils import TrackedObject


class TrackHistory:
    """Maintain per-object track history for hazard logic."""

    def __init__(self, history_len: int = 10):
        self.history_len = history_len
        self._history: dict[int, Deque[TrackedObject]] = {}

    def update(self, objects: list[TrackedObject]) -> dict[int, deque[TrackedObject]]:
        """Add new detections to history, return current history dict."""
        for obj in objects:
            tid = obj.id
            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.history_len)
            self._history[tid].append(obj)
        return dict(self._history)

    def get_history(self, obj_id: int) -> deque[TrackedObject]:
        """Get history for one object."""
        return self._history.get(obj_id, deque())
