from __future__ import annotations

import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "dismisser-matplotlib"),
)

import cv2
import mediapipe as mp
import numpy as np


class CameraPreviewWindow:
    """Small camera preview with FaceMesh and magnified eye diagnostics."""

    WINDOW_NAME = "Dismisser Camera Preview"
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

    def __init__(
        self,
        screen_size: tuple[int, int],
        width: int = 420,
        margin: int = 18,
    ) -> None:
        self.screen_width, self.screen_height = screen_size
        self.width = width
        self.margin = margin
        self._created = False
        self._face_mesh = mp.solutions.face_mesh
        self._drawing_utils = mp.solutions.drawing_utils
        self._face_spec = self._drawing_utils.DrawingSpec(
            color=(90, 210, 255),
            thickness=1,
            circle_radius=1,
        )
        self._eye_spec = self._drawing_utils.DrawingSpec(
            color=(70, 255, 120),
            thickness=1,
            circle_radius=1,
        )
        self._iris_spec = self._drawing_utils.DrawingSpec(
            color=(255, 120, 60),
            thickness=1,
            circle_radius=1,
        )

    def show(self, frame: np.ndarray, face_landmarks) -> bool:
        preview = self._render(frame, face_landmarks)
        self._refresh_window_state()
        self._ensure_window(preview.shape[1], preview.shape[0])
        cv2.imshow(self.WINDOW_NAME, preview)
        key = cv2.waitKey(1) & 0xFF
        return key in (ord("q"), 27)

    def close(self) -> None:
        if self._created:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
            except cv2.error:
                pass
            self._created = False

    def _ensure_window(self, width: int, height: int) -> None:
        if self._created:
            return
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, width, height)
        x = self.margin
        y = max(0, self.screen_height - height - self.margin)
        cv2.moveWindow(self.WINDOW_NAME, x, y)
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            try:
                cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
        self._created = True

    def _refresh_window_state(self) -> None:
        if not self._created or not hasattr(cv2, "WND_PROP_VISIBLE"):
            return
        try:
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._created = False
        except cv2.error:
            self._created = False

    def _render(self, frame: np.ndarray, face_landmarks) -> np.ndarray:
        annotated = frame.copy()
        if face_landmarks is None:
            cv2.putText(
                annotated,
                "No face",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (40, 40, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            self._draw_face_landmarks(annotated, face_landmarks)

        preview = self._resize_width(annotated, self.width)
        if face_landmarks is not None:
            eye_view = self._render_eye_inset(frame, face_landmarks, preview.shape[1])
            if eye_view is not None:
                face_box = self._scaled_landmark_box(
                    face_landmarks,
                    frame.shape[1],
                    frame.shape[0],
                    preview.shape[1],
                    preview.shape[0],
                )
                self._paste_inset(preview, eye_view, face_box)
        return preview

    def _draw_face_landmarks(self, image: np.ndarray, face_landmarks) -> None:
        self._drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self._face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._face_spec,
        )
        self._drawing_utils.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self._face_mesh.FACEMESH_LEFT_EYE | self._face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._eye_spec,
        )
        iris_connections = getattr(self._face_mesh, "FACEMESH_IRISES", frozenset())
        if iris_connections:
            self._drawing_utils.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=iris_connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._iris_spec,
            )

    def _render_eye_inset(
        self,
        frame: np.ndarray,
        face_landmarks,
        preview_width: int,
    ) -> np.ndarray | None:
        frame_height, frame_width = frame.shape[:2]
        points = self._landmark_points(face_landmarks, frame_width, frame_height)
        indexes = self._eye_indexes()
        selected = [points[index] for index in indexes if index in points]
        if not selected:
            return None

        target_width = max(160, int(preview_width * 0.48))
        target_height = max(72, int(target_width * 0.42))
        crop_box = self._eye_crop_box(
            selected,
            frame_width,
            frame_height,
            aspect_ratio=target_width / target_height,
        )
        if crop_box is None:
            return None

        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None
        inset = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        scale_x = target_width / max(1, x2 - x1)
        scale_y = target_height / max(1, y2 - y1)
        self._draw_eye_detail(inset, points, x1, y1, scale_x, scale_y)
        return inset

    def _landmark_points(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
    ) -> dict[int, tuple[int, int]]:
        return {
            index: (int(landmark.x * frame_width), int(landmark.y * frame_height))
            for index, landmark in enumerate(face_landmarks.landmark)
        }

    def _eye_indexes(self) -> set[int]:
        indexes = {
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_OUTER,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
            *self.LEFT_IRIS,
            *self.RIGHT_IRIS,
        }
        for connections in (
            self._face_mesh.FACEMESH_LEFT_EYE,
            self._face_mesh.FACEMESH_RIGHT_EYE,
            getattr(self._face_mesh, "FACEMESH_IRISES", frozenset()),
        ):
            for start, end in connections:
                indexes.add(start)
                indexes.add(end)
        return indexes

    def _eye_crop_box(
        self,
        points: list[tuple[int, int]],
        frame_width: int,
        frame_height: int,
        aspect_ratio: float,
    ) -> tuple[int, int, int, int] | None:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if not xs or not ys:
            return None

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        box_width = width * 1.7
        box_height = height * 3.2
        if box_width / max(1.0, box_height) < aspect_ratio:
            box_width = box_height * aspect_ratio
        else:
            box_height = box_width / aspect_ratio

        x1 = int(max(0, center_x - box_width / 2.0))
        y1 = int(max(0, center_y - box_height / 2.0))
        x2 = int(min(frame_width, center_x + box_width / 2.0))
        y2 = int(min(frame_height, center_y + box_height / 2.0))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _draw_eye_detail(
        self,
        image: np.ndarray,
        points: dict[int, tuple[int, int]],
        crop_x: int,
        crop_y: int,
        scale_x: float,
        scale_y: float,
    ) -> None:
        self._draw_connections(
            image,
            points,
            self._face_mesh.FACEMESH_LEFT_EYE | self._face_mesh.FACEMESH_RIGHT_EYE,
            crop_x,
            crop_y,
            scale_x,
            scale_y,
            color=(60, 255, 120),
        )
        self._draw_connections(
            image,
            points,
            getattr(self._face_mesh, "FACEMESH_IRISES", frozenset()),
            crop_x,
            crop_y,
            scale_x,
            scale_y,
            color=(255, 130, 50),
        )
        for index in (
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_INNER,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_OUTER,
        ):
            self._draw_point(image, points, index, crop_x, crop_y, scale_x, scale_y, (40, 255, 255), 3)
        for index in (*self.LEFT_IRIS, *self.RIGHT_IRIS):
            self._draw_point(image, points, index, crop_x, crop_y, scale_x, scale_y, (255, 80, 40), 2)

    def _draw_connections(
        self,
        image: np.ndarray,
        points: dict[int, tuple[int, int]],
        connections,
        crop_x: int,
        crop_y: int,
        scale_x: float,
        scale_y: float,
        color: tuple[int, int, int],
    ) -> None:
        for start, end in connections:
            if start not in points or end not in points:
                continue
            cv2.line(
                image,
                self._scaled_point(points[start], crop_x, crop_y, scale_x, scale_y),
                self._scaled_point(points[end], crop_x, crop_y, scale_x, scale_y),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_point(
        self,
        image: np.ndarray,
        points: dict[int, tuple[int, int]],
        index: int,
        crop_x: int,
        crop_y: int,
        scale_x: float,
        scale_y: float,
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        if index not in points:
            return
        cv2.circle(
            image,
            self._scaled_point(points[index], crop_x, crop_y, scale_x, scale_y),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    def _scaled_point(
        self,
        point: tuple[int, int],
        crop_x: int,
        crop_y: int,
        scale_x: float,
        scale_y: float,
    ) -> tuple[int, int]:
        return (
            int((point[0] - crop_x) * scale_x),
            int((point[1] - crop_y) * scale_y),
        )

    def _scaled_landmark_box(
        self,
        face_landmarks,
        frame_width: int,
        frame_height: int,
        preview_width: int,
        preview_height: int,
    ) -> tuple[int, int, int, int] | None:
        points = [
            (landmark.x * frame_width, landmark.y * frame_height)
            for landmark in face_landmarks.landmark
            if 0.0 <= landmark.x <= 1.0 and 0.0 <= landmark.y <= 1.0
        ]
        if not points:
            return None
        scale_x = preview_width / frame_width
        scale_y = preview_height / frame_height
        xs = [point[0] * scale_x for point in points]
        ys = [point[1] * scale_y for point in points]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def _paste_inset(
        self,
        preview: np.ndarray,
        inset: np.ndarray,
        avoid_box: tuple[int, int, int, int] | None,
    ) -> None:
        margin = 8
        border = 2
        inset_height, inset_width = inset.shape[:2]
        x, y = self._choose_inset_position(
            preview.shape[1],
            preview.shape[0],
            inset_width,
            inset_height,
            margin,
            avoid_box,
        )
        cv2.rectangle(
            preview,
            (x - border, y - border),
            (x + inset_width + border, y + inset_height + border),
            (8, 8, 8),
            -1,
        )
        cv2.rectangle(
            preview,
            (x - border, y - border),
            (x + inset_width + border, y + inset_height + border),
            (240, 240, 240),
            1,
        )
        preview[y : y + inset_height, x : x + inset_width] = inset

    def _choose_inset_position(
        self,
        preview_width: int,
        preview_height: int,
        inset_width: int,
        inset_height: int,
        margin: int,
        avoid_box: tuple[int, int, int, int] | None,
    ) -> tuple[int, int]:
        candidates = (
            (margin, margin),
            (preview_width - inset_width - margin, margin),
            (margin, preview_height - inset_height - margin),
            (preview_width - inset_width - margin, preview_height - inset_height - margin),
        )
        candidates = tuple(
            (max(margin, x), max(margin, y))
            for x, y in candidates
        )
        if avoid_box is None:
            return candidates[2]

        avoid_center_x = (avoid_box[0] + avoid_box[2]) / 2.0
        avoid_center_y = (avoid_box[1] + avoid_box[3]) / 2.0

        def score(candidate: tuple[int, int]) -> tuple[int, float]:
            x, y = candidate
            candidate_box = (x, y, x + inset_width, y + inset_height)
            overlap = self._rect_intersection_area(candidate_box, avoid_box)
            candidate_center_x = x + inset_width / 2.0
            candidate_center_y = y + inset_height / 2.0
            distance_sq = (
                (candidate_center_x - avoid_center_x) ** 2
                + (candidate_center_y - avoid_center_y) ** 2
            )
            return overlap, -distance_sq

        return min(candidates, key=score)

    def _rect_intersection_area(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> int:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _resize_width(self, image: np.ndarray, width: int) -> np.ndarray:
        height = int(image.shape[0] * (width / image.shape[1]))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
