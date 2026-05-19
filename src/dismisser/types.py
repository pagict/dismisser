from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GazePoint:
    """Normalized screen-ish gaze coordinates, where 0..1 maps left..right/top..bottom."""

    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class HeadPose:
    """Lightweight normalized head pose features derived from face landmarks."""

    yaw: float
    pitch: float
    roll: float


@dataclass(frozen=True)
class RawGazeSample:
    raw_x: float
    raw_y: float
    head_pose: HeadPose
