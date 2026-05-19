# Dismisser

Dismisser 是一个桌面端 MVP：使用摄像头估计用户在屏幕上的注视位置，当用户持续看向通知区域时，自动尝试点灭或取消引人注意的通知。

当前实现优先服务快速验证，但代码按可替换层次拆分，方便后续把 Python 模块替换成 C/C++、Objective-C/Swift 或 Win32/UIAutomation 原生实现。

## 功能概览

- 使用 MediaPipe FaceMesh 读取眼部和脸部关键点。
- 使用 OpenCV `solvePnP` 估计头部姿态。
- 通过校准数据把眼球 raw gaze 和头部姿态映射到屏幕坐标。
- 使用 FOXTracker/OpenTrack Accela 风格的 deadzone + smoothing 稳定输出。
- 使用全屏透明 overlay 显示当前注视点和通知目标区域。
- 默认 dry-run；只有加 `--enable-actions` 才会移动鼠标或尝试取消通知。

## 安装

建议使用 Python 3.12 或 3.11。MediaPipe 通常不支持最新 Python 版本。

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

macOS 需要给终端或 Python 进程授权：

- Camera：摄像头权限
- Accessibility：仅在使用 `--enable-actions` 时需要

## 运行

推荐先采集校准数据：

```bash
.venv/bin/python calibrate.py --grid 5x5 --samples-per-point 30
```

然后运行主程序：

```bash
.venv/bin/python run.py
```

如果安装 entry points 后，也可以运行：

```bash
dismisser
dismisser-calibrate
```

主程序启动时会自动读取最新的 `calibration_samples/gaze-calibration-*.jsonl`。没有校准数据时，会回退到启发式 gaze 映射。

## 常用参数

```bash
.venv/bin/python calibrate.py --grid 7x5 --samples-per-point 30
.venv/bin/python run.py --calibration calibration_samples/gaze-calibration-xxxx.jsonl
.venv/bin/python run.py --no-gaze-filter
.venv/bin/python run.py --gaze-filter-deadzone 0.006
.venv/bin/python run.py --enable-actions
```

## 快捷键

主程序 overlay：

- `q` 或 `Esc`：退出
- `c`：把当前注视状态记录为 neutral
- `r`：重置 neutral

校准程序：

- `Enter`：采集当前红点
- `q` 或 `Esc`：退出

## 架构交接

核心路径：

- `src/dismisser/main.py`：CLI 参数入口
- `src/dismisser/app.py`：主流程编排
- `src/dismisser/gaze.py`：摄像头帧到 gaze/head pose
- `src/dismisser/calibration.py`：校准数据采集
- `src/dismisser/calibration_model.py`：线性/二阶校准模型
- `src/dismisser/gaze_filter.py`：Accela 风格输出滤波
- `src/dismisser/overlay.py`：屏幕透明 overlay
- `src/dismisser/platform_actions.py`：平台通知取消动作

后续替换方向：

- 用原生或模型更强的 gaze tracker 替换 `MediaPipeGazeTracker`。
- 用 macOS Accessibility/UserNotifications 或 Windows UIAutomation 替换鼠标模拟。
- 把校准点绘制也迁移到同一个真实屏幕 overlay，以减少 OpenCV fullscreen 的坐标误差。

## 当前限制

- 普通摄像头 gaze 精度有限，强依赖摄像头位置、光照和眼镜反光。
- 校准数据绑定当前用户、屏幕、摄像头位置。
- macOS 当前通过指针拖拽模拟取消通知。
- Windows 当前只是占位式 tray hover/escape 行为，需要后续原生实现。
