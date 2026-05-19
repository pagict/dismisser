from __future__ import annotations

import os
import tempfile
from typing import Protocol

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "dismisser-matplotlib"),
)

import cv2
import mediapipe as mp
import numpy as np

from dismisser.types import GazePoint, HeadPose, RawGazeSample
from dismisser.calibration_model import CalibrationModel


class GazeTracker(Protocol):
    def estimate(self, frame: np.ndarray) -> GazePoint | None:
        ...

    def set_neutral(self) -> None:
        ...

    def reset_neutral(self) -> None:
        ...

    def close(self) -> None:
        ...


class MediaPipeGazeTracker:
    """Fast MVP gaze estimate using FaceMesh iris landmarks.

    This is intentionally heuristic. The interface is the part to preserve when
    replacing it with calibrated native eye tracking.
    """

    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    LEFT_IRIS = (468, 469, 470, 471)
    RIGHT_IRIS = (473, 474, 475, 476)
    NOSE_TIP = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 152

    def __init__(
        self,
        smoothing: float = 0.25,
        calibration: CalibrationModel | None = None,
    ) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._neutral: tuple[float, float] | None = None
        self._neutral_pose: HeadPose | None = None
        self._pending_neutral: tuple[float, float] | None = None
        self._pending_neutral_pose: HeadPose | None = None
        self._smoothed: tuple[float, float] | None = None
        self._last_raw: tuple[float, float] | None = None
        self._last_sample: RawGazeSample | None = None
        self._smoothing = smoothing
        self._calibration = calibration

    def estimate(self, frame: np.ndarray) -> GazePoint | None:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(image)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        raw_x = self._eye_ratio(
            landmarks,
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.LEFT_IRIS,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_IRIS,
        )
        raw_y = self._vertical_ratio(
            landmarks,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
            self.LEFT_IRIS,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
            self.RIGHT_IRIS,
        )
        head_pose = self._head_pose(landmarks, frame.shape)
        self._last_raw = (raw_x, raw_y)
        self._last_sample = RawGazeSample(raw_x, raw_y, head_pose)

        if self._pending_neutral is not None:
            self._neutral = self._pending_neutral
            self._neutral_pose = self._pending_neutral_pose
            self._pending_neutral = None
            self._pending_neutral_pose = None

        if self._calibration is not None:
            raw_x, raw_y = self._calibration.map_raw(raw_x, raw_y, head_pose)
        elif self._neutral is not None:
            pose_dx = head_pose.yaw - (self._neutral_pose.yaw if self._neutral_pose else 0.0)
            pose_dy = head_pose.pitch - (self._neutral_pose.pitch if self._neutral_pose else 0.0)
            raw_x = 0.5 + ((raw_x - self._neutral[0]) - pose_dx * 0.22) * 2.4
            raw_y = 0.5 + ((raw_y - self._neutral[1]) + pose_dy * 0.16) * 2.4
        else:
            raw_x = raw_x - head_pose.yaw * 0.10
            raw_y = raw_y + head_pose.pitch * 0.08

        raw_x = float(np.clip(raw_x, 0.0, 1.0))
        raw_y = float(np.clip(raw_y, 0.0, 1.0))
        if self._smoothed is None:
            self._smoothed = (raw_x, raw_y)
        else:
            old_x, old_y = self._smoothed
            alpha = self._smoothing
            self._smoothed = (
                old_x * (1.0 - alpha) + raw_x * alpha,
                old_y * (1.0 - alpha) + raw_y * alpha,
            )

        return GazePoint(self._smoothed[0], self._smoothed[1], confidence=0.7)

    def set_neutral(self) -> None:
        if self._last_sample is not None:
            self._pending_neutral = (self._last_sample.raw_x, self._last_sample.raw_y)
            self._pending_neutral_pose = self._last_sample.head_pose
            self._smoothed = None

    def reset_neutral(self) -> None:
        self._neutral = None
        self._neutral_pose = None
        self._pending_neutral = None
        self._pending_neutral_pose = None
        self._smoothed = None

    def last_raw(self) -> tuple[float, float] | None:
        return self._last_raw

    def last_sample(self) -> RawGazeSample | None:
        return self._last_sample

    def close(self) -> None:
        self._mesh.close()

    def _eye_ratio(
        self,
        landmarks: list,
        left_outer: int,
        left_inner: int,
        left_iris: tuple[int, ...],
        right_inner: int,
        right_outer: int,
        right_iris: tuple[int, ...],
    ) -> float:
        left = self._ratio_between(landmarks, left_outer, left_inner, left_iris, axis="x")
        right = self._ratio_between(landmarks, right_inner, right_outer, right_iris, axis="x")
        return (left + right) / 2.0

    def _vertical_ratio(
        self,
        landmarks: list,
        left_top: int,
        left_bottom: int,
        left_iris: tuple[int, ...],
        right_top: int,
        right_bottom: int,
        right_iris: tuple[int, ...],
    ) -> float:
        left = self._ratio_between(landmarks, left_top, left_bottom, left_iris, axis="y")
        right = self._ratio_between(landmarks, right_top, right_bottom, right_iris, axis="y")
        return (left + right) / 2.0

    def _ratio_between(
        self,
        landmarks: list,
        start_index: int,
        end_index: int,
        iris_indexes: tuple[int, ...],
        axis: str,
    ) -> float:
        start = getattr(landmarks[start_index], axis)
        end = getattr(landmarks[end_index], axis)
        iris = float(np.mean([getattr(landmarks[index], axis) for index in iris_indexes]))
        denom = end - start
        if abs(denom) < 1e-6:
            return 0.5
        return float(np.clip((iris - start) / denom, 0.0, 1.0))

    def _head_pose(self, landmarks: list, frame_shape: tuple[int, ...]) -> HeadPose:
        frame_height, frame_width = frame_shape[:2]
        image_points = np.array(
            [
                self._pixel_point(landmarks, self.NOSE_TIP, frame_width, frame_height),
                self._pixel_point(landmarks, self.CHIN, frame_width, frame_height),
                self._pixel_point(landmarks, self.LEFT_EYE_OUTER, frame_width, frame_height),
                self._pixel_point(landmarks, self.RIGHT_EYE_OUTER, frame_width, frame_height),
                self._pixel_point(landmarks, self.MOUTH_LEFT, frame_width, frame_height),
                self._pixel_point(landmarks, self.MOUTH_RIGHT, frame_width, frame_height),
            ],
            dtype=np.float64,
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )
        focal_length = float(frame_width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2.0],
                [0.0, focal_length, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)
        ok, rotation_vector, _translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            return self._rotation_to_head_pose(rotation_matrix)

        left_eye = self._point_midpoint(landmarks, self.LEFT_EYE_OUTER, self.LEFT_EYE_INNER)
        right_eye = self._point_midpoint(landmarks, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER)
        roll = float(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        return HeadPose(0.0, 0.0, roll)

    def _point(self, landmarks: list, index: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array([landmark.x, landmark.y], dtype=float)

    def _point_midpoint(self, landmarks: list, first: int, second: int) -> np.ndarray:
        return (self._point(landmarks, first) + self._point(landmarks, second)) / 2.0

    def _pixel_point(
        self,
        landmarks: list,
        index: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        landmark = landmarks[index]
        return landmark.x * frame_width, landmark.y * frame_height

    def _rotation_to_head_pose(self, rotation_matrix: np.ndarray) -> HeadPose:
        sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        singular = sy < 1e-6
        if not singular:
            pitch = float(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
            yaw = float(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        else:
            pitch = float(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
            yaw = float(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = 0.0
        return HeadPose(yaw, pitch, roll)
