This project is modified from https://github.com/discoverse-dev, changed by Yidan Ma only for final year project. Basicly only used the gs_playground simulator.


# FR5 单相机仿真与模仿学习 Demo

这是 `gs_playground` 中的 FR5 机械臂示例工程，包含：

- FR5 + Robotiq 85 夹爪的 MuJoCo/MotrixSim 仿真场景。
- Astra Pro Plus 单 RGB 相机采集与相机外参标定。
- 三色胶带随机摆放、按颜色顺序套入 `myd_part1` 的仿真任务。
- 行为克隆 imitation learning 数据生成、训练、批量闭环验证。
- 可选真机连接脚本，用于 FR5 控制器和 Robotiq 夹爪联动。

本文件面向复现实验和二次开发，不包含个人项目说明。

## 1. 环境

推荐使用已有的 `uv` 工作流：

```bash
cd ~/gs_playground
uv sync
```

常用依赖包括：

- Python 3.10
- numpy / torch / matplotlib
- OpenCV
- MotrixSim
- Orbbec `pyorbbecsdk`，仅在使用 Astra 相机时需要

如果只运行仿真数据生成、训练和验证，不连接真实相机和真实机械臂，可以先跳过 Orbbec 和 Fairino 真机 SDK。

## 2. 目录

主要文件位于：

```text
demo/fr5_demo/
├── assets/fr5/mjmodel.xml              # FR5 仿真场景
├── configs/fr5_table_task.json         # 任务参数、初始位姿、目标和胶带配置
├── configs/astra_camera.json           # Astra 相机内外参
├── fr5_sim_tape_pick_place.py          # 仿真示教数据生成
├── fr5_train_bc.py                     # 单模型 BC 训练
├── fr5_train_all_bc_models.py          # 多模型批量训练
├── fr5_policy_rollout.py               # 单策略闭环验证
├── fr5_evaluate_all_policies.py        # 多策略批量闭环验证
├── fr5_show_real_targets.py            # 真实相机目标检测可视化
└── notebooks/
    ├── fr5_bc_training_history_analysis.ipynb
    └── fr5_policy_batch_eval_analysis.ipynb
```

## 3. 仿真示教数据生成

生成仿真 imitation learning episode：

```bash
uv run python demo/fr5_demo/fr5_sim_tape_pick_place.py \
  --episodes 120 \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --hz 10 \
  --attach-assist \
  --no-visualize
```

需要观察仿真窗口时使用：

```bash
uv run python demo/fr5_demo/fr5_sim_tape_pick_place.py \
  --episodes 1 \
  --visualize \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --hz 10 \
  --attach-assist
```

数据默认保存到：

```text
demo/fr5_demo/data/il_demos/
```

## 4. 训练模型

批量训练当前配置中的多个 BC 模型：

```bash
uv run python demo/fr5_demo/fr5_train_all_bc_models.py
```

训练结果默认保存到：

```text
demo/fr5_demo/data/il_policies/
```

常见输出：

```text
fr5_bc_state_mlp_tape_bc_v1.pt
fr5_bc_cnn_small_tape_bc_v1.pt
fr5_bc_spatial_softmax_tape_bc_v1.pt
*.history.json
```

训练曲线可用 notebook 查看：

```text
demo/fr5_demo/notebooks/fr5_bc_training_history_analysis.ipynb
```

## 5. 批量闭环验证

推荐先运行 `assist` 条件，并比较 `phase_guard=off` 与 `phase_guard=soft`：

```bash
uv run python demo/fr5_demo/fr5_evaluate_all_policies.py \
  --policy-pattern 'fr5_bc_*_tape_bc_v1.pt' \
  --conditions assist \
  --num-seeds 10 \
  --max-steps 250 \
  --hz 10 \
  --lookahead-frames 3 \
  --max-action-deg 45 \
  --max-gripper-delta 0.05 \
  --sim-rgb-source visual \
  --phase-guards soft off \
  --policy-ablations normal \
  --action-ema 0.25 \
  --max-action-step-deg 12 \
  --run-name fr5_eval_ablation_v1
```

输出目录：

```text
demo/fr5_demo/data/policy_rollouts/batch_eval/fr5_eval_ablation_v1/
```

结果分析 notebook：

```text
demo/fr5_demo/notebooks/fr5_policy_batch_eval_analysis.ipynb
```

指标含义：

- `success_rate`：闭环任务成功率。
- `place_error_mean_m`：最终放置 XY 误差均值。
- `clip_fraction_mean`：动作被限幅比例。
- `min_tcp_z_mean_m`：执行期间 TCP 最低高度均值。
- `phase_guard=off`：纯策略闭环推理。
- `phase_guard=soft`：在策略输出基础上加入软阶段约束。

## 6. Astra 相机

只查看 Astra 实时 RGB：

```bash
uv run python demo/fr5_demo/camera_capture_orbbec.py
```

无窗口采集少量图片：

```bash
uv run python demo/fr5_demo/camera_capture_orbbec.py \
  --no-display \
  --max-frames 30 \
  --save-every 1
```

相机配置写入：

```text
demo/fr5_demo/configs/astra_camera.json
```

## 7. 动态标定

如果使用真实 Astra 与真实 FR5，需要先完成相机到机械臂底座坐标系的外参标定。推荐使用动态 marker 标定流程：

```bash
uv run python demo/fr5_demo/fr5_dynamic_marker_calibration.py collect \
  --robot-ip 192.168.58.2 \
  --marker-id 0 \
  --marker-length 0.08 \
  --real-rgb-fps 10 \
  --vel 10 \
  --acc 10 \
  --ovl 20 \
  --execute-real \
  --reset-errors
```

求解并写入相机配置：

```bash
uv run python demo/fr5_demo/fr5_dynamic_marker_calibration.py solve \
  --write-camera
```

## 8. 真实目标检测可视化

在真实相机画面中显示胶带和 `myd_part1` 的估计位置：

```bash
uv run python demo/fr5_demo/fr5_show_real_targets.py \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --config demo/fr5_demo/configs/fr5_table_task.json \
  --active-color red \
  --real-rgb-source live
```

如果桌面反光严重，建议先采集空桌背景，再使用背景差分或调整白色胶带的阈值参数。

## 9. 真机控制

真机相关脚本默认需要：

- FR5 控制器网络可达，例如 `192.168.58.2`。
- Fairino RPC 服务正常。
- Robotiq 夹爪通过 USB/串口连接，例如 `/dev/ttyUSB0`。
- 操作前确认急停、限位和工作空间安全。

移动到初始位姿：

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --vel 10 \
  --acc 10 \
  --ovl 20
```

键盘触发真实抓取：

```bash
uv run python demo/fr5_demo/fr5_real_keyboard_grasp.py \
  --robot-ip 192.168.58.2 \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --config demo/fr5_demo/configs/fr5_table_task.json \
  --active-color red \
  --hz 30 \
  --speed-percent 5 \
  --execute-real \
  --execute-gripper \
  --gripper-port /dev/ttyUSB0
```

键位：

- `a`：根据当前检测到的胶带位置开始一集。
- `b`：采集空桌背景，用于反光抑制。
- `s`：停止当前集并返回初始位姿。
- `q` / `Esc`：安全退出。

## 10. 常见问题

**找不到 rollout 日志**

先运行 `fr5_evaluate_all_policies.py`，然后在 notebook 中确认 `SELECTED_RUN_NAME` 指向正确 run 名。

**Astra 报 frame queue full**

通常是相机帧产生速度高于处理速度。可以降低显示频率、关闭不必要窗口，或使用脚本中的 latest-frame/drain 逻辑。

**真实相机识别白色胶带不稳定**

白色胶带容易受桌面反光影响。建议使用空桌背景差分、降低曝光、改变光源角度，或给白色胶带增加非白色边缘标记。

**soft 成功但 off 失败**

这说明纯策略闭环容易出现长时序误差累积。`soft` 结果应解释为“策略 + 软阶段约束”的部署效果，而不是纯端到端 BC 效果。

## 11. 安全提示

真实机械臂运行前请先在仿真中验证轨迹，并保持低速。任何真实抓取脚本都应在可随时急停的条件下运行。
