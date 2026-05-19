from __future__ import annotations

from dataclasses import dataclass
import time

from dismisser.config import PlatformTarget
from dismisser.types import GazePoint


@dataclass(frozen=True)
class AttentionEvent:
    target_name: str
    dwell_seconds: float
    gaze: GazePoint


class AttentionDetector:
    """Detects when gaze dwells inside the notification zone."""

    def __init__(
        self,
        platform: PlatformTarget,
        dwell_seconds: float,
        cooldown_seconds: float,
    ) -> None:
        self.platform = platform
        self.dwell_seconds = dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self._entered_at: float | None = None
        self._last_triggered_at = 0.0

    def update(self, gaze: GazePoint | None) -> AttentionEvent | None:
        now = time.monotonic()
        if gaze is None or not self._in_target(gaze):
            self._entered_at = None
            return None

        if self._entered_at is None:
            self._entered_at = now
            return None

        dwell = now - self._entered_at
        if dwell < self.dwell_seconds:
            return None
        if now - self._last_triggered_at < self.cooldown_seconds:
            return None

        self._last_triggered_at = now
        self._entered_at = now
        return AttentionEvent(self._target_name(), dwell, gaze)

    def _in_target(self, gaze: GazePoint) -> bool:
        if self.platform == PlatformTarget.MAC:
            return gaze.x >= 0.72 and gaze.y <= 0.28
        if self.platform == PlatformTarget.WINDOWS:
            return gaze.x >= 0.72 and gaze.y >= 0.72
        return False

    def _target_name(self) -> str:
        if self.platform == PlatformTarget.MAC:
            return "macOS top-right notification area"
        return "Windows bottom-right notification area"
