"""Debouncing and console output for hazard events."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .utils import HazardEvent


class HazardNotifier:
    """Per-(hazard_type, object_id) debouncing and console output."""

    def __init__(self, cooldown_s: float = 0.5):
        self.cooldown_s = cooldown_s
        self._last: dict[tuple[str, int | None], float] = {}

    def _key(self, event: "HazardEvent") -> tuple[str, int | None]:
        return (event.type, event.object_id)

    def should_report(self, event: "HazardEvent", current_time: float) -> bool:
        """True if event passes cooldown."""
        k = self._key(event)
        last = self._last.get(k, -999.0)
        if current_time - last >= self.cooldown_s:
            self._last[k] = current_time
            return True
        return False

    def report(self, event: "HazardEvent") -> None:
        """Print hazard to console in structured format."""
        details = event.details.copy()
        parts = [f"conf={event.confidence:.2f}"]
        if event.object_id is not None:
            parts.insert(0, f"id={event.object_id}")
        for k, v in details.items():
            parts.append(f"{k}={v}")
        fmt = "HAZARD: {} ({})".format(event.type, " ".join(str(p) for p in parts))
        print(fmt)
