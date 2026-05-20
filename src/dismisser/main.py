from __future__ import annotations

import argparse
from pathlib import Path

from dismisser.config import AppConfig, PlatformTarget, default_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dismiss flashing notifications after gaze dwells on the notification area."
    )
    parser.add_argument(
        "--platform",
        choices=[target.value for target in PlatformTarget],
        default=None,
        help="Override platform target. Defaults to current OS.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--dwell-seconds",
        type=float,
        default=1.0,
        help="Seconds gaze must remain in target zone before triggering.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=2.5,
        help="Minimum seconds between dismissal attempts.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without the full-screen gaze overlay.",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="Show a small camera diagnostics window with face and eye landmarks.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally mirror the camera frame.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Use a specific calibration JSONL file.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("calibration_samples"),
        help="Directory to search for the newest calibration JSONL file.",
    )
    parser.add_argument(
        "--no-gaze-filter",
        action="store_true",
        help="Disable Accela-style gaze output smoothing.",
    )
    parser.add_argument(
        "--gaze-filter-smoothing",
        type=float,
        default=0.055,
        help="Accela-style gaze smoothing time constant.",
    )
    parser.add_argument(
        "--gaze-filter-deadzone",
        type=float,
        default=0.006,
        help="Normalized gaze deadzone for output stabilization.",
    )
    parser.add_argument(
        "--enable-ui-snap",
        action="store_true",
        help="Snap gaze to nearby OS accessibility/UI Automation elements.",
    )
    parser.add_argument(
        "--ui-snap-radius-px",
        type=int,
        default=80,
        help="Maximum pixel distance from gaze to an accessibility element before snapping.",
    )
    parser.add_argument(
        "--ui-snap-refresh-seconds",
        type=float,
        default=0.10,
        help="Minimum seconds between accessibility hit-test refreshes.",
    )
    parser.add_argument(
        "--enable-actions",
        action="store_true",
        help="Actually move the pointer and attempt dismissal.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    platform = PlatformTarget(args.platform) if args.platform else default_platform()
    config = AppConfig(
        platform=platform,
        camera_index=args.camera,
        dwell_seconds=args.dwell_seconds,
        cooldown_seconds=args.cooldown_seconds,
        preview=not args.no_preview,
        camera_preview=args.camera_preview,
        mirror_camera=not args.no_mirror,
        calibration_path=args.calibration,
        calibration_dir=args.calibration_dir,
        gaze_filter=not args.no_gaze_filter,
        gaze_filter_smoothing=args.gaze_filter_smoothing,
        gaze_filter_deadzone=args.gaze_filter_deadzone,
        ui_snap=args.enable_ui_snap,
        ui_snap_radius_px=args.ui_snap_radius_px,
        ui_snap_refresh_seconds=args.ui_snap_refresh_seconds,
        enable_actions=args.enable_actions,
    )
    from dismisser.app import DismisserApp

    return DismisserApp(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
