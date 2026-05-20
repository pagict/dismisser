# Runtime Adaptive Calibration Design

本文记录运行阶段动态校准的设计方向。当前不实现该能力；后续需要做动态校准时，应优先按本文约束推进。

## 背景

现有运行链路是：

1. `MediaPipeGazeTracker` 从摄像头帧中提取 iris ratios 和 head pose。
2. `CalibrationModel` 把 raw gaze 和 head pose 映射到归一化屏幕坐标。
3. `Accela2DGazeFilter` 做 deadzone smoothing。
4. `GazeOverlay` 显示 gaze 点和通知目标区域。
5. `AttentionDetector` 根据 dwell time 触发 `NotificationDismisser`。

离线校准会受到摄像头位置、坐姿、光照、眼镜反光、用户疲劳和 head pose 漂移影响。运行一段时间后，模型可能出现系统性偏差，例如用户实际想看 A 点，但系统识别到 A 上方的 B 点。

用户可能会自然补偿这个偏差：略微向下看、低头或调整坐姿，让识别点更接近真实关注位置。问题在于，单看这个动作无法判断它是“用户在补偿校准误差”，还是“用户确实想继续看下方内容”。

因此动态校准不能把用户动作本身当作校准意图。

## 设计原则

- 动态校准必须是锚点驱动，而不是动作驱动。
- 不直接在线改写离线 `CalibrationModel`。
- 初版只做 session 内的保守修正，确认稳定后再考虑持久化。
- 更新必须慢、可限幅、可重置、可观测。
- 校准层不应该理解通知业务语义；业务侧只提供高置信度 anchor。
- `AttentionDetector` 仍负责 dwell/触发判断，不应混入底层 gaze/head-pose 估计逻辑。

## 建议插入位置

新增一个运行期自适应层，放在离线校准之后、滤波之前：

```text
MediaPipeGazeTracker
  -> CalibrationModel
  -> AdaptiveCalibrationLayer
  -> Accela2DGazeFilter
  -> GazeOverlay / AttentionDetector
```

原因：

- `CalibrationModel` 的输出已经是屏幕归一化坐标，适合做偏移修正。
- 放在 `Accela2DGazeFilter` 前，可以让滤波器稳定最终 corrected gaze，而不是稳定一个已知偏差点。
- 不破坏原有离线校准数据格式，也方便用开关完全关闭动态校准。

## 核心模型

初版不要在线重训二次模型。建议从最小 bias 模型开始：

```text
corrected_x = calibrated_x + bias_x
corrected_y = calibrated_y + bias_y
```

后续可以逐步升级：

1. 全局 `dx/dy`。
2. 按屏幕区域分块的 `dx/dy`。
3. 按 head pose bucket 的 `dx/dy`。
4. 低阶偏移场，例如基于位置和 head pose 的小型线性修正。

不要一开始做复杂在线模型，否则很难判断错误来自 gaze tracker、离线校准、滤波、业务 anchor，还是动态层自身。

## Anchor 来源

动态校准只从 anchor 学习。anchor 表示“系统有额外证据认为用户真实意图接近这个屏幕点或区域”。

### 高置信度 anchor

- 用户显式点击某个点。
- 用户按键确认当前目标。
- `--enable-actions` 模式下，系统尝试 notification dismiss 且后续结果表明动作成功。
- 用户手动把鼠标移动到某个目标并完成操作。

这些 anchor 可以用于实际更新。

### 中置信度 anchor

- 用户在通知区域附近稳定 dwell，且随后触发 dismiss。
- UI snap 找到明确可交互控件，用户 gaze 长时间稳定在控件附近。
- 同一目标附近多次出现稳定 gaze，再由业务事件确认。

这些 anchor 初期只建议用于日志和分析；确认有效后再允许小权重更新。

### 低置信度证据

- 用户略向下看。
- 用户低头。
- gaze 点从 B 移到 A。
- 用户在某个方向有补偿趋势。

这些只能作为辅助上下文，不能单独触发校准更新。

## A/B 偏差场景判断

典型场景：

- 用户真实想看 A 点。
- 模型输出落在 A 上方 B 点。
- 用户略向下看或低头，让输出点接近 A。

错误做法：

```text
用户向下看 -> 认为用户在校准 -> 立即向下修正模型
```

正确做法：

```text
模型先落在 B
用户调整后落到 A 附近
A 附近存在当前任务目标
用户在 A 附近稳定 dwell
后续动作或系统结果验证 A 是意图目标
多次重复出现同类偏差
=> 认为存在系统性向上 bias
=> 慢速更新动态偏移
```

也就是说，动作不是判断依据；动作、目标语义、稳定性、重复性和结果反馈共同组成证据链。

## 更新门控

只有同时满足以下条件时才允许更新：

- 当前 gaze 稳定，没有快速扫视。
- head pose 稳定，没有明显低头/转头过程中的过渡帧。
- anchor 置信度达到阈值。
- 当前 residual 没有超过异常值阈值。
- 最近一段时间的 residual 方向和大小具有一致性。
- 用户没有刚按下 `c`/`r` 等会改变 neutral 状态的快捷键。
- 动态校准没有达到最大偏移限制。

建议保留 hard off 开关，便于调试：

```bash
.venv/bin/python run.py --no-adaptive-calibration
```

或反过来默认关闭：

```bash
.venv/bin/python run.py --enable-adaptive-calibration
```

初期建议默认关闭。

## 更新公式

对高置信度 anchor，可以计算 residual：

```text
residual = anchor_point - calibrated_point
```

然后用很小学习率更新：

```text
bias = (1 - alpha) * bias + alpha * residual
```

建议初值：

- `alpha`: `0.01` 到 `0.03`
- `max_bias`: 屏幕归一化坐标 `0.03` 到 `0.08`
- `min_stable_ms`: `500` 到 `1000`
- `min_repeated_anchors`: `3`

这些值只能作为起点，必须通过实际日志验证。

## Head Pose 处理

head pose 是动态校准里最容易污染模型的变量。

例如用户低头时，gaze 映射偏差可能和正常坐姿完全不同。如果把低头时的 residual 更新到全局 bias，正常坐姿会变差。

建议：

- 初版记录 `head_yaw`、`head_pitch`、`head_roll`，但只用稳定坐姿附近的数据更新全局 bias。
- 如果需要支持坐姿变化，再引入 head pose bucket。
- bucket 之间不要共享强更新，只做弱平滑。
- head pose 变化过快时只记录日志，不更新。

## 日志与可观测性

实现前应先补日志格式，确保可以离线判断动态校准是否有效。

每个候选 anchor 至少记录：

- timestamp
- raw gaze feature
- calibrated point
- corrected point
- filtered point
- anchor point or anchor rect
- anchor source
- anchor confidence
- residual
- current bias
- head_yaw/head_pitch/head_roll
- gaze stability metrics
- whether update was applied
- rejection reason when skipped

日志应足够回答：

- 动态校准是否持续朝一个方向漂移？
- 更新是否集中发生在某个 head pose？
- 更新是否真的降低 anchor residual？
- 是否有单个错误 anchor 把 bias 拉偏？
- 关闭动态层后结果是否更稳定？

## MVP 实施步骤

1. 新增 `AdaptiveCalibrationLayer`，默认 disabled。
2. 输入 `calibrated_point`、`head_pose` 和可选 `anchor`，输出 `corrected_point`。
3. 只实现 session 内全局 `dx/dy`。
4. 加最大偏移限制和 reset 能力。
5. 先只记录候选 anchor 和 residual，不应用更新。
6. 从高置信度 anchor 开始启用小权重更新。
7. 增加 CLI 开关和日志选项。
8. 用实际运行日志比较 disabled/enabled 的 residual 和 dismiss 成功率。
9. 只有在日志证明收益稳定后，再考虑持久化 bias。

## 不建议做的事

- 不要根据“用户向某个方向看”直接更新校准。
- 不要把动态校准写回原始 calibration jsonl。
- 不要让 `AttentionDetector` 直接修改 calibration model。
- 不要一开始做复杂在线二次拟合。
- 不要在没有 anchor 结果反馈的情况下持久化动态修正。
- 不要让单次 dismiss 成功大幅改变模型。

## 验证标准

动态校准实现后，至少需要验证：

- 关闭动态层时行为与当前版本一致。
- 启用但无 anchor 时输出不漂移。
- 错误 anchor 被限幅和门控阻断。
- 多个一致高置信度 anchor 能缓慢修正系统性偏差。
- head pose 变化不会污染正常坐姿 bias。
- reset 后 corrected gaze 回到离线校准输出。
- `--enable-actions` 仍是唯一会移动指针或尝试通知取消的模式。

## 结论

运行阶段动态校准是可行的，但应被设计成保守的在线 bias 修正层。它需要依赖可靠 anchor，而不是试图直接识别用户是否“正在校准”。用户的补偿动作只能作为辅助证据；真正触发学习的应该是稳定、重复、可验证的意图锚点。
