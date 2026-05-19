from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dismisser.types import HeadPose


@dataclass(frozen=True)
class CalibrationModel:
    path: Path
    coefficients_x: np.ndarray
    coefficients_y: np.ndarray
    sample_count: int
    uses_head_pose: bool
    feature_mode: str

    def map_raw(
        self,
        raw_x: float,
        raw_y: float,
        head_pose: HeadPose | None = None,
    ) -> tuple[float, float]:
        features = self._features(raw_x, raw_y, head_pose)
        x = float(np.clip(features @ self.coefficients_x, 0.0, 1.0))
        y = float(np.clip(features @ self.coefficients_y, 0.0, 1.0))
        return x, y

    def _features(
        self,
        raw_x: float,
        raw_y: float,
        head_pose: HeadPose | None,
    ) -> np.ndarray:
        if self.uses_head_pose:
            pose = head_pose or HeadPose(0.0, 0.0, 0.0)
            if self.feature_mode == "quadratic":
                return np.array(
                    [
                        1.0,
                        raw_x,
                        raw_y,
                        raw_x * raw_x,
                        raw_x * raw_y,
                        raw_y * raw_y,
                        pose.yaw,
                        pose.pitch,
                        pose.roll,
                        raw_x * pose.yaw,
                        raw_y * pose.pitch,
                    ],
                    dtype=float,
                )
            return np.array([1.0, raw_x, raw_y, pose.yaw, pose.pitch, pose.roll], dtype=float)
        if self.feature_mode == "quadratic":
            return np.array(
                [1.0, raw_x, raw_y, raw_x * raw_x, raw_x * raw_y, raw_y * raw_y],
                dtype=float,
            )
        return np.array([1.0, raw_x, raw_y], dtype=float)


def load_latest_calibration(directory: Path) -> CalibrationModel | None:
    files = sorted(directory.glob("gaze-calibration-*.jsonl"))
    if not files:
        return None
    for path in reversed(files):
        model = load_calibration(path)
        if model is not None:
            return model
    return None


def load_calibration(path: Path) -> CalibrationModel | None:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if _is_valid_record(record):
                records.append(record)

    if len(records) < 3:
        return None

    uses_head_pose = all(_has_head_pose(record) for record in records)
    feature_mode = _select_feature_mode(len(records), uses_head_pose)
    features = np.array(
        [_record_features(record, uses_head_pose, feature_mode) for record in records],
        dtype=float,
    )
    target_x = np.array([record["target_x"] for record in records], dtype=float)
    target_y = np.array([record["target_y"] for record in records], dtype=float)
    coefficients_x = np.linalg.lstsq(features, target_x, rcond=None)[0]
    coefficients_y = np.linalg.lstsq(features, target_y, rcond=None)[0]
    return CalibrationModel(
        path,
        coefficients_x,
        coefficients_y,
        len(records),
        uses_head_pose,
        feature_mode,
    )


def _is_valid_record(record: dict) -> bool:
    keys = ("raw_x", "raw_y", "target_x", "target_y")
    return all(isinstance(record.get(key), int | float) for key in keys)


def _has_head_pose(record: dict) -> bool:
    keys = ("head_yaw", "head_pitch", "head_roll")
    return all(isinstance(record.get(key), int | float) for key in keys)


def _select_feature_mode(record_count: int, uses_head_pose: bool) -> str:
    required = 11 if uses_head_pose else 6
    return "quadratic" if record_count >= required else "linear"


def _record_features(record: dict, uses_head_pose: bool, feature_mode: str) -> list[float]:
    raw_x = record["raw_x"]
    raw_y = record["raw_y"]
    if uses_head_pose:
        yaw = record["head_yaw"]
        pitch = record["head_pitch"]
        roll = record["head_roll"]
        if feature_mode == "quadratic":
            return [
                1.0,
                raw_x,
                raw_y,
                raw_x * raw_x,
                raw_x * raw_y,
                raw_y * raw_y,
                yaw,
                pitch,
                roll,
                raw_x * yaw,
                raw_y * pitch,
            ]
        return [1.0, raw_x, raw_y, yaw, pitch, roll]
    if feature_mode == "quadratic":
        return [1.0, raw_x, raw_y, raw_x * raw_x, raw_x * raw_y, raw_y * raw_y]
    return [1.0, raw_x, raw_y]
