from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from dismisser.types import GazePoint


@dataclass(frozen=True)
class Accela2DConfig:
    smoothing: float = 0.055
    deadzone: float = 0.006


class Accela2DGazeFilter:
    """FOXTracker/OpenTrack Accela-style output filter for normalized gaze.

    Small changes inside the deadzone are held still. Larger changes move at a
    speed proportional to distance, preserving quick large movements while
    damping jitter around the current output.
    """

    def __init__(self, config: Accela2DConfig) -> None:
        self.config = config
        self._last_output: np.ndarray | None = None
        self._last_time: float | None = None

    def reset(self) -> None:
        self._last_output = None
        self._last_time = None

    def update(self, gaze: GazePoint | None) -> GazePoint | None:
        now = time.monotonic()
        if gaze is None:
            self._last_time = now
            return None

        current = np.array([gaze.x, gaze.y], dtype=float)
        if self._last_output is None or self._last_time is None:
            self._last_output = current
            self._last_time = now
            return gaze

        dt = min(max(now - self._last_time, 1.0 / 240.0), 0.05)
        self._last_time = now
        delta = current - self._last_output
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.deadzone:
            return GazePoint(
                float(self._last_output[0]),
                float(self._last_output[1]),
                gaze.confidence,
            )

        effective = max(distance - self.config.deadzone, 0.0)
        direction = delta / max(distance, 1e-9)
        velocity = effective / max(self.config.smoothing, 1e-6)
        step = direction * velocity * dt
        if float(np.linalg.norm(step)) > distance:
            self._last_output = current
        else:
            self._last_output = self._last_output + step
        self._last_output = np.clip(self._last_output, 0.0, 1.0)
        return GazePoint(float(self._last_output[0]), float(self._last_output[1]), gaze.confidence)
