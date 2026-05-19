from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import platform as platform_module
from pathlib import Path


class PlatformTarget(str, Enum):
    MAC = "mac"
    WINDOWS = "windows"


@dataclass(frozen=True)
class AppConfig:
    platform: PlatformTarget
    camera_index: int = 0
    dwell_seconds: float = 1.0
    cooldown_seconds: float = 2.5
    preview: bool = True
    mirror_camera: bool = True
    calibration_path: Path | None = None
    calibration_dir: Path = Path("calibration_samples")
    gaze_filter: bool = True
    gaze_filter_smoothing: float = 0.055
    gaze_filter_deadzone: float = 0.006
    enable_actions: bool = False


def default_platform() -> PlatformTarget:
    system = platform_module.system().lower()
    if system == "darwin":
        return PlatformTarget.MAC
    if system == "windows":
        return PlatformTarget.WINDOWS
    raise RuntimeError(f"Unsupported platform for notification dismissal: {system}")
