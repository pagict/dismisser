# Repository Guidelines

## Project Structure

- `src/dismisser/`: application package.
- `run.py`: source-tree launcher for the main gaze overlay app.
- `calibrate.py`: source-tree launcher for calibration collection.
- `tests/fixtures/`: small calibration fixtures used for smoke checks.
- `calibration_samples/`: local user calibration output; keep untracked.

## Core Flow

The runtime pipeline is:

1. `MediaPipeGazeTracker` reads webcam frames and extracts iris ratios plus head pose.
2. Head pose uses MediaPipe landmarks and OpenCV `solvePnP`.
3. `CalibrationModel` maps raw gaze and head pose to normalized screen coordinates.
4. `Accela2DGazeFilter` applies FOXTracker/OpenTrack-style deadzone smoothing.
5. `GazeOverlay` draws the screen-space gaze point and target notification zone.
6. `AttentionDetector` triggers `NotificationDismisser` after dwell time.

Keep these boundaries intact when replacing Python pieces with native code.

## Commands

Install:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

Run from source without refreshing entry points:

```bash
.venv/bin/python run.py
.venv/bin/python calibrate.py --grid 5x5 --samples-per-point 30
```

Verify:

```bash
python3.12 -m compileall src calibrate.py run.py
PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from dismisser.calibration_model import load_calibration; m=load_calibration(Path("tests/fixtures/sample_calibration.jsonl")); print(m.sample_count, m.feature_mode)'
```

## Development Notes

- Default to source launchers because existing local `.venv` metadata may be stale or non-writable.
- Do not commit `.venv/`, `__pycache__/`, `.egg-info/`, or `calibration_samples/`.
- New calibration files should include `head_yaw`, `head_pitch`, and `head_roll`.
- A 5x5 or denser calibration grid is expected for `quadratic` calibration.
- macOS overlay uses PyObjC/AppKit; Windows fallback currently uses Tk.
- `--enable-actions` is the only mode that moves the pointer or attempts notification dismissal.
