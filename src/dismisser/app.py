from __future__ import annotations

import time

import cv2

from dismisser.attention import AttentionDetector
from dismisser.calibration_model import load_calibration, load_latest_calibration
from dismisser.config import AppConfig
from dismisser.gaze_filter import Accela2DConfig, Accela2DGazeFilter
from dismisser.gaze import MediaPipeGazeTracker
from dismisser.overlay import GazeOverlay
from dismisser.platform_actions import PyAutoGuiNotificationDismisser


class DismisserApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.gaze_tracker: MediaPipeGazeTracker | None = None
        self.attention = AttentionDetector(
            config.platform,
            dwell_seconds=config.dwell_seconds,
            cooldown_seconds=config.cooldown_seconds,
        )
        self.dismisser = PyAutoGuiNotificationDismisser(
            config.platform,
            enable_actions=config.enable_actions,
        )
        self.gaze_filter = (
            Accela2DGazeFilter(
                Accela2DConfig(
                    smoothing=config.gaze_filter_smoothing,
                    deadzone=config.gaze_filter_deadzone,
                )
            )
            if config.gaze_filter
            else None
        )
        self._stop_requested = False

    def run(self) -> int:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            print(f"Unable to open camera index {self.config.camera_index}")
            return 2
        calibration = self._load_calibration()
        if calibration is not None:
            print(
                "Loaded gaze calibration "
                f"path={calibration.path} samples={calibration.sample_count} "
                f"head_pose={'yes' if calibration.uses_head_pose else 'no'} "
                f"features={calibration.feature_mode}"
            )
        else:
            print("No gaze calibration loaded; using heuristic gaze mapping")
        self.gaze_tracker = MediaPipeGazeTracker(calibration=calibration)

        print(
            "Dismisser running "
            f"platform={self.config.platform.value} "
            f"actions={'enabled' if self.config.enable_actions else 'dry-run'}"
        )
        print("Overlay keys: q/Esc=quit, c=calibrate neutral, r=reset calibration")

        try:
            if self.config.preview:
                overlay = GazeOverlay(
                    self.config.platform,
                    on_quit=self._request_stop,
                    on_capture_neutral=self._capture_neutral,
                    on_reset_neutral=self._reset_neutral,
                )
                return overlay.run(lambda: self._process_frame(capture, overlay.update_gaze))
            return self._run_headless(capture)
        finally:
            capture.release()
            if self.gaze_tracker is not None:
                self.gaze_tracker.close()
            cv2.destroyAllWindows()

    def _run_headless(self, capture) -> int:
        while not self._stop_requested:
            exit_code = self._process_frame(capture, lambda _gaze: None)
            if exit_code is not None:
                return exit_code
        return 0

    def _process_frame(self, capture, update_gaze) -> int | None:
        if self._stop_requested:
            return 0
        ok, frame = capture.read()
        if not ok:
            print("Camera frame read failed")
            return 3
        if self.config.mirror_camera:
            frame = cv2.flip(frame, 1)

        gaze = self.gaze_tracker.estimate(frame) if self.gaze_tracker is not None else None
        if self.gaze_filter is not None:
            gaze = self.gaze_filter.update(gaze)
        update_gaze(gaze)
        event = self.attention.update(gaze)
        if event is not None:
            result = self.dismisser.dismiss_attention_target()
            print(
                f"{time.strftime('%H:%M:%S')} target={event.target_name} "
                f"gaze=({event.gaze.x:.2f},{event.gaze.y:.2f}) "
                f"dwell={event.dwell_seconds:.2f}s result={result.message}"
            )
        return None

    def _load_calibration(self):
        if self.config.calibration_path is not None:
            return load_calibration(self.config.calibration_path)
        return load_latest_calibration(self.config.calibration_dir)

    def _request_stop(self) -> None:
        self._stop_requested = True

    def _capture_neutral(self) -> None:
        if self.gaze_tracker is not None:
            self.gaze_tracker.set_neutral()
            if self.gaze_filter is not None:
                self.gaze_filter.reset()
            print("Neutral gaze captured")

    def _reset_neutral(self) -> None:
        if self.gaze_tracker is not None:
            self.gaze_tracker.reset_neutral()
            if self.gaze_filter is not None:
                self.gaze_filter.reset()
            print("Neutral gaze reset")
