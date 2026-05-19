# Dismisser

Dismisser is a desktop MVP that watches webcam gaze, estimates the user's screen-space focus, and dismisses attention-catching notifications after the user dwells on the notification area.

The project is Python-based for fast iteration, but the code is split into replaceable layers:

- `GazeTracker`: webcam frame input and gaze estimation.
- `AttentionDetector`: dwell-time logic for deciding when the user is looking at the notification area.
- `NotificationDismisser`: platform-specific action backend.
- `DismisserApp`: orchestration and gaze overlay UI.

This keeps the upper-level flow stable when the webcam/gaze layer is later replaced by a faster native C/C++ backend, or when macOS/Windows notification control moves to Objective-C/Swift or Win32/UIAutomation.

## Install

Use Python 3.12 or 3.11. MediaPipe may not support the newest Python releases.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

On macOS, grant Terminal or the Python executable:

- Camera permission
- Accessibility permission, only needed when running with `--enable-actions`

## Run

Dry-run mode, with a full-screen transparent gaze overlay:

```bash
dismisser
```

If entry points are unavailable or stale, run directly from the source tree:

```bash
.venv/bin/python run.py
```

On startup, `dismisser` automatically loads the newest `calibration_samples/gaze-calibration-*.jsonl` file when one exists. Without calibration data, it falls back to the built-in heuristic mapping.

Actually move the pointer and attempt dismissal:

```bash
dismisser --enable-actions
```

Collect gaze calibration samples:

```bash
dismisser-calibrate
```

Without installing entry points, run directly from source:

```bash
.venv/bin/python calibrate.py
```

The calibration collector displays a full-screen red point on a 5x5 grid by default. Look at the point and press `Enter`; it saves averaged raw gaze samples to `calibration_samples/*.jsonl`.

New calibration files include lightweight head-pose features (`head_yaw`, `head_pitch`, `head_roll`) so runtime gaze mapping can compensate for head movement. Older calibration files still load, but they do not include head-pose correction; rerun `calibrate.py` after this change for better tracking.

With enough samples, runtime calibration uses a quadratic feature model instead of a simple affine fit. The default 5x5 grid provides enough points for this. You can use a denser grid when the notification area still feels inaccurate:

```bash
.venv/bin/python calibrate.py --grid 7x5 --samples-per-point 30
```

Recommended first run:

```bash
.venv/bin/python calibrate.py --grid 5x5 --samples-per-point 30
.venv/bin/python run.py
```

Useful flags:

```bash
dismisser --platform mac --dwell-seconds 0.8
dismisser --platform windows --camera 1
dismisser --calibration calibration_samples/gaze-calibration-20260519-140000.jsonl
dismisser --calibration-dir calibration_samples
dismisser --no-mirror
dismisser --no-gaze-filter
dismisser --gaze-filter-deadzone 0.006
dismisser --no-preview --enable-actions
```

Keyboard controls in overlay:

- `q` or `Esc`: quit
- `c`: capture the current gaze as neutral center
- `r`: reset calibration

Calibration controls:

- `Enter`: capture the current red point
- `q` or `Esc`: quit

## Current Behavior

macOS target area: top-right notification/banner area.

Windows target area: bottom-right system tray/notification icon area.

The MVP uses heuristic gaze estimation from FaceMesh iris landmarks. It is good enough to validate the interaction loop, but it is not precision eye tracking. For production, replace `dismisser.gaze.MediaPipeGazeTracker` with a calibrated native tracker and replace `dismisser.platform_actions.PyAutoGuiNotificationDismisser` with native OS automation.

The head-pose and stabilization path follows the same broad architecture as FOXTracker: estimate face pose separately from raw eye direction, remap through calibration, then stabilize the output with an Accela-style deadzone/smoothing filter. This MVP uses MediaPipe landmarks plus OpenCV `solvePnP` instead of FOXTracker's heavier FSA-Net/ONNX/Qt pipeline.

## Handoff Notes

- Main entry point: `src/dismisser/main.py`.
- Main orchestration: `src/dismisser/app.py`.
- Calibration collection: `src/dismisser/calibration.py`.
- Calibration fitting: `src/dismisser/calibration_model.py`.
- Gaze estimation: `src/dismisser/gaze.py`.
- Screen overlay: `src/dismisser/overlay.py`.
- Pointer-based dismissal backend: `src/dismisser/platform_actions.py`.

Known limitations:

- Webcam gaze is approximate and depends heavily on camera placement and lighting.
- Current calibration is per-user and per-camera-position.
- macOS dismissal currently simulates a top-right swipe; a production version should use native notification APIs or Accessibility automation.
- Windows behavior is a placeholder hover/escape action and needs Win32/UIAutomation work.
