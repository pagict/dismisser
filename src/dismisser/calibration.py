from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np

from dismisser.gaze import MediaPipeGazeTracker
from dismisser.types import RawGazeSample


@dataclass(frozen=True)
class CalibrationConfig:
    camera_index: int = 0
    mirror_camera: bool = True
    output_dir: Path = Path("calibration_samples")
    samples_per_point: int = 18
    grid_columns: int = 5
    grid_rows: int = 5
    margin: float = 0.06


@dataclass(frozen=True)
class TargetPoint:
    name: str
    x: float
    y: float


def grid_targets(columns: int, rows: int, margin: float) -> tuple[TargetPoint, ...]:
    columns = max(2, columns)
    rows = max(2, rows)
    margin = float(np.clip(margin, 0.0, 0.25))
    xs = np.linspace(margin, 1.0 - margin, columns)
    ys = np.linspace(margin, 1.0 - margin, rows)
    center_y = rows // 2
    ordered: list[TargetPoint] = []
    for row_index, y in enumerate(ys):
        row_points = [
            TargetPoint(f"r{row_index + 1}_c{column_index + 1}", float(x), float(y))
            for column_index, x in enumerate(xs)
        ]
        if abs(row_index - center_y) % 2 == 1:
            row_points.reverse()
        ordered.extend(row_points)
    return tuple(ordered)


class CalibrationCollector:
    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self.window_name = "Dismisser Calibration"
        self.targets = grid_targets(config.grid_columns, config.grid_rows, config.margin)

    def run(self) -> int:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            print(f"Unable to open camera index {self.config.camera_index}")
            return 2

        tracker = MediaPipeGazeTracker(smoothing=1.0)
        target_sample_count = max(1, self.config.samples_per_point)
        raw_history: deque[RawGazeSample] = deque(maxlen=target_sample_count)
        output_path = self._output_path()
        print(f"Saving calibration samples to {output_path}")
        print("Look at the red point and press Enter to start sampling. Press q or Esc to quit.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        target_index = 0
        sampling = False
        try:
            while target_index < len(self.targets):
                ok, frame = capture.read()
                if not ok:
                    print("Camera frame read failed")
                    return 3
                if self.config.mirror_camera:
                    frame = cv2.flip(frame, 1)

                tracker.estimate(frame)
                sample = tracker.last_sample()
                if sampling and sample is not None:
                    raw_history.append(sample)
                    if len(raw_history) >= target_sample_count:
                        self._append_sample(output_path, self.targets[target_index], raw_history)
                        print(f"Captured {self.targets[target_index].name}")
                        raw_history.clear()
                        target_index += 1
                        sampling = False
                        continue

                canvas = self._draw_target(
                    self.targets[target_index],
                    len(raw_history),
                    target_index,
                    sampling,
                )
                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    return 0
                if key in (10, 13) and not sampling:
                    raw_history.clear()
                    sampling = True

            print(f"Calibration collection complete: {output_path}")
            return 0
        finally:
            capture.release()
            tracker.close()
            cv2.destroyAllWindows()

    def _draw_target(
        self,
        target: TargetPoint,
        buffered_samples: int,
        target_index: int,
        sampling: bool,
    ) -> np.ndarray:
        canvas = np.zeros((900, 1440, 3), dtype=np.uint8)
        height, width = canvas.shape[:2]
        x = int(target.x * width)
        y = int(target.y * height)
        target_samples = max(1, self.config.samples_per_point)
        remaining_samples = max(0, target_samples - buffered_samples)
        progress = 1.0 - (remaining_samples / target_samples)
        cv2.circle(canvas, (x, y), 18, (0, 0, 255), -1)
        cv2.circle(canvas, (x, y), 30, (0, 0, 180), 3)
        bar_width = 96
        bar_height = 8
        bar_x1 = x - bar_width // 2
        bar_y1 = y + 44
        bar_x2 = bar_x1 + bar_width
        bar_y2 = bar_y1 + bar_height
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (45, 45, 45), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (0, 120, 0), 1)
        cv2.rectangle(
            canvas,
            (bar_x1, bar_y1),
            (int(bar_x1 + bar_width * progress), bar_y2),
            (0, 220, 0),
            -1,
        )
        cv2.putText(
            canvas,
            f"n={remaining_samples}",
            (bar_x1, bar_y2 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 220, 0),
            2,
        )
        cv2.putText(
            canvas,
            "Sampling..." if sampling else "Focus red point, press Enter to sample",
            (40, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (230, 230, 230),
            2,
        )
        cv2.putText(
            canvas,
            f"{target.name}  {target_index + 1}/{len(self.targets)}  remaining:{remaining_samples}",
            (40, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (180, 180, 180),
            2,
        )
        cv2.putText(
            canvas,
            "q/Esc quits",
            (40, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (140, 140, 140),
            2,
        )
        return canvas

    def _append_sample(
        self,
        output_path: Path,
        target: TargetPoint,
        raw_history: deque[RawGazeSample],
    ) -> None:
        samples = list(raw_history)
        raw_x = float(np.mean([sample.raw_x for sample in samples]))
        raw_y = float(np.mean([sample.raw_y for sample in samples]))
        head_yaw = float(np.mean([sample.head_pose.yaw for sample in samples]))
        head_pitch = float(np.mean([sample.head_pose.pitch for sample in samples]))
        head_roll = float(np.mean([sample.head_pose.roll for sample in samples]))
        record = {
            "timestamp": time.time(),
            "target_name": target.name,
            "target_x": target.x,
            "target_y": target.y,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "head_roll": head_roll,
            "raw_std_x": float(np.std([sample.raw_x for sample in samples])),
            "raw_std_y": float(np.std([sample.raw_y for sample in samples])),
            "head_std_yaw": float(np.std([sample.head_pose.yaw for sample in samples])),
            "head_std_pitch": float(np.std([sample.head_pose.pitch for sample in samples])),
            "head_std_roll": float(np.std([sample.head_pose.roll for sample in samples])),
            "sample_count": len(samples),
            "mirror_camera": self.config.mirror_camera,
            "grid_columns": self.config.grid_columns,
            "grid_rows": self.config.grid_rows,
            "grid_margin": self.config.margin,
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _output_path(self) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return self.config.output_dir / f"gaze-calibration-{timestamp}.jsonl"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Collect gaze calibration samples.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally mirror the camera frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_samples"),
        help="Directory for JSONL calibration output.",
    )
    parser.add_argument(
        "--samples-per-point",
        type=int,
        default=18,
        help="Recent eye samples to average when Enter is pressed.",
    )
    parser.add_argument(
        "--grid",
        default="5x5",
        help="Calibration grid size, for example 5x5 or 7x5.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.06,
        help="Normalized screen margin for calibration points.",
    )
    args = parser.parse_args()
    columns, rows = _parse_grid(args.grid)
    config = CalibrationConfig(
        camera_index=args.camera,
        mirror_camera=not args.no_mirror,
        output_dir=args.output_dir,
        samples_per_point=args.samples_per_point,
        grid_columns=columns,
        grid_rows=rows,
        margin=args.margin,
    )
    return CalibrationCollector(config).run()


def _parse_grid(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("*", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise SystemExit(f"Invalid grid size: {value}. Expected format like 5x5.")
    try:
        columns = int(parts[0])
        rows = int(parts[1])
    except ValueError as exc:
        raise SystemExit(f"Invalid grid size: {value}. Expected integers like 5x5.") from exc
    if columns < 2 or rows < 2:
        raise SystemExit("Calibration grid must be at least 2x2.")
    return columns, rows


if __name__ == "__main__":
    raise SystemExit(main())
