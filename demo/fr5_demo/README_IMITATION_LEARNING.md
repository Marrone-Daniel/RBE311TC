# FR5 模仿学习操作 Pipeline

只包含操作提示和命令。默认工作目录：

```bash
cd ~/gs_playground
```

## 1. 基础检查

检查 FR5 网络端口：

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py \
  --robot-ip 192.168.58.2 \
  --diagnose-only
```

检查 Astra：

```bash
uv run python demo/fr5_demo/camera_capture_orbbec.py \
  --no-display \
  --max-frames 30 \
  --save-every 10
```

检查手柄按键编号：

```bash
uv run python demo/fr5_demo/fr5_gamepad_inspect.py
```

检查夹爪，优先试 `pyrobotiq`：

```bash
uv run python demo/fr5_demo/fr5_gripper_diagnose.py \
  --port /dev/ttyUSB0 \
  --backend pyrobotiq \
  --slave-id 9 \
  --sweep
```

如果失败，再试 raw Modbus：

```bash
uv run python demo/fr5_demo/fr5_gripper_diagnose.py \
  --port /dev/ttyUSB0 \
  --backend raw \
  --slave-id 9 \
  --timeout 1.0 \
  --retries 3 \
  --sweep
```

## 2. 回到初始位姿

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --vel 50 \
  --acc 50 \
  --ovl 80
```

如控制器有错误码：

```bash
uv run python demo/fr5_demo/fr5_cancel_motion.py --servo-end
```

必要时：

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --reset-errors \
  --vel 50 \
  --acc 50 \
  --ovl 80
```

## 3. 手柄键位

```text
Left stick       TCP X/Y
Right stick Y    TCP Z
LB / RB          TCP yaw
LT / RT          TCP pitch
D-pad up/down    TCP roll
X                close gripper while held
Y                open gripper while held
A                print current gripper target
B                end current episode, return to initial_qpos, then wait for next input
Ctrl+C           stop safely and save valid current episode
```

如果 X/Y/B 映射不对，先运行：

```bash
uv run python demo/fr5_demo/fr5_gamepad_inspect.py
```

然后在采集命令里加类似参数：

```bash
--button-x 2 --button-y 3 --button-b 1
```

## 4. 采集真机示教数据

推荐命令：

```bash
uv run python demo/fr5_demo/fr5_record_demonstration.py \
  --robot-ip 192.168.58.2 \
  --xbox-teleop \
  --execute-gripper \
  --gripper-backend pyrobotiq \
  --gripper-port /dev/ttyUSB0 \
  --gripper-speed 255 \
  --gripper-force 150 \
  --gripper-step 0.06 \
  --gripper-send-period 0.03 \
  --hz 10 \
  --return-vel 50 \
  --return-acc 50 \
  --return-ovl 80 \
  --max-frames 300 \
  --gamepad-debug
```

如果暂时不控制真实夹爪：

```bash
uv run python demo/fr5_demo/fr5_record_demonstration.py \
  --robot-ip 192.168.58.2 \
  --xbox-teleop \
  --hz 10 \
  --return-vel 50 \
  --return-acc 50 \
  --return-ovl 80 \
  --gamepad-debug
```

如果要限制采集时长：

```bash
--duration 60
```

如果要限制每条 episode 最大帧数：

```bash
--max-frames 300
```

## 5. 检查采集结果

列出有效 episode：

```bash
find demo/fr5_demo/data/il_demos -maxdepth 2 -name states.npz -print
```

查看空 episode：

```bash
find demo/fr5_demo/data/il_demos -maxdepth 1 -type d -empty -print
```

删除空 episode：

```bash
find demo/fr5_demo/data/il_demos -maxdepth 1 -type d -empty -delete
```

检查单条 episode：

```bash
uv run python demo/fr5_demo/fr5_replay_demonstration.py \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000 \
  --check-only
```

可视化重放：

```bash
uv run python demo/fr5_demo/fr5_replay_demonstration.py \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000
```

如果当前机器没有 CUDA，关闭 sim 3DGS 屏幕：

```bash
uv run python demo/fr5_demo/fr5_replay_demonstration.py \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000 \
  --no-sim-gs-screen
```

## 6. 训练 BC 策略

训练全部有效 episode：

```bash
uv run python demo/fr5_demo/fr5_train_bc.py \
  --data-dir demo/fr5_demo/data/il_demos \
  --model-type cnn_small \
  --epochs 30 \
  --batch-size 32 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --out demo/fr5_demo/data/il_policies/fr5_bc_last.pt
```

可选模型：

```text
cnn_small    默认小 CNN + 关节状态
cnn_medium   更大 CNN，容量更高，容易对比是否过拟合
late_fusion  轻量视觉编码后和状态后融合
state_mlp    只用关节/夹爪状态，不用图像，作为单相机方法的消融基线
```

一次性训练全部模型：

```bash
uv run python demo/fr5_demo/fr5_train_all_bc_models.py \
  --data-dir demo/fr5_demo/data/il_demos \
  --epochs 30 \
  --batch-size 32 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --run-name tape_bc_v1
```

输出：

```text
demo/fr5_demo/data/il_policies/fr5_bc_cnn_small_tape_bc_v1.pt
demo/fr5_demo/data/il_policies/fr5_bc_cnn_medium_tape_bc_v1.pt
demo/fr5_demo/data/il_policies/fr5_bc_late_fusion_tape_bc_v1.pt
demo/fr5_demo/data/il_policies/fr5_bc_state_mlp_tape_bc_v1.pt
demo/fr5_demo/data/il_policies/fr5_bc_model_comparison_tape_bc_v1.json
```

只训练指定 episode：

```bash
uv run python demo/fr5_demo/fr5_train_bc.py \
  --data-dir demo/fr5_demo/data/il_demos \
  --episodes episode_YYYYmmdd_HHMMSS_000 episode_YYYYmmdd_HHMMSS_001 \
  --model-type cnn_medium \
  --epochs 30 \
  --batch-size 32 \
  --out demo/fr5_demo/data/il_policies/fr5_bc_last.pt
```

快速过拟合测试：

```bash
uv run python demo/fr5_demo/fr5_train_bc.py \
  --data-dir demo/fr5_demo/data/il_demos \
  --episodes episode_YYYYmmdd_HHMMSS_000 \
  --epochs 80 \
  --batch-size 16 \
  --val-ratio 0.0 \
  --out demo/fr5_demo/data/il_policies/fr5_bc_overfit.pt
```

## 7. 离线验证策略

无窗口检查：

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000 \
  --no-window
```

有窗口检查：

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000
```

没有 CUDA 时：

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000 \
  --no-window
```

## 8. 现场输入，仿真镜像，不动真机

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --max-steps 100 \
  --hz 5 \
  --max-action-deg 0.5 \
  --max-gripper-delta 0.03
```

## 9. 真机部署，机械臂，不动夹爪

先用小步长：

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --max-steps 60 \
  --hz 5 \
  --max-action-deg 0.3 \
  --max-gripper-delta 0.02
```

更保守：

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --max-steps 30 \
  --hz 3 \
  --max-action-deg 0.2 \
  --max-gripper-delta 0.01
```

## 10. 真机部署，机械臂 + 夹爪

先确认夹爪诊断通过。

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --execute-gripper \
  --gripper-port /dev/ttyUSB0 \
  --max-steps 60 \
  --hz 5 \
  --max-action-deg 0.3 \
  --max-gripper-delta 0.02
```

## 11. 可选：生成仿真示教数据

当前 3DGS 渲染需要 CUDA。

默认轨迹：

```bash
uv run python demo/fr5_demo/fr5_generate_sim_demonstration.py \
  --hz 10 \
  --gripper-closure 0.0
```

从真机 episode 的轨迹生成：

```bash
uv run python demo/fr5_demo/fr5_generate_sim_demonstration.py \
  --replay-qpos demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS_000/replay_qpos.npz \
  --source-dt 0.1 \
  --hz 10
```

## 12. 可选：Real2Sim 资产生成

静态场景绑定，FR5/桌子/固定底座使用已有 MuJoCo 模型，3DGS 只作为视觉层：

```bash
uv run python -m real2sim.examples.run_real2sim \
  --input_dir ./real2sim_input \
  --output_dir ./real2sim_output \
  --config real2sim/examples/config_example.yaml
```

动态抓取目标才使用 Astra RGB-D：

```bash
uv run python -m real2sim.examples.prepare_real2sim_input \
  --output-dir ./real2sim_input \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --fps 10 \
  --color-format mjpg \
  --align-mode none \
  --mask-mode roi \
  --object-name target_object \
  --run-real2sim
```

无窗口采集，全图作为 mask：

```bash
uv run python -m real2sim.examples.prepare_real2sim_input \
  --output-dir ./real2sim_input \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --fps 10 \
  --color-format mjpg \
  --align-mode none \
  --no-display \
  --mask-mode full
```

使用已有 mask：

```bash
uv run python -m real2sim.examples.prepare_real2sim_input \
  --output-dir ./real2sim_input \
  --camera-config demo/fr5_demo/configs/astra_camera.json \
  --fps 10 \
  --color-format mjpg \
  --align-mode none \
  --mask-mode existing \
  --mask-file path/to/mask.png
```

生成后的输入目录：

```text
real2sim_input/
  rgb.png              # Astra RGB
  depth.png            # Astra depth, uint16 mm
  mask.png             # white=object
  intrinsics.yaml      # Orbbec SDK RGB intrinsics
  camera_pose.yaml     # from astra_camera.json
  config.yaml          # object and export settings
```

只运行 pipeline：

```bash
uv run python -m real2sim.examples.run_real2sim \
  --input_dir ./real2sim_input \
  --output_dir ./real2sim_output \
  --config ./real2sim_input/config.yaml
```

查看结果：

```bash
find real2sim_output -type f | sort
```

重点检查：

```text
real2sim_output/sim/scene_gs_binding.yaml
real2sim_output/debug/scene_binding_report.md
real2sim_output/debug/projected_overlay.png
real2sim_output/debug/pointcloud_debug.ply
real2sim_output/sim/mujoco_object.xml
real2sim_output/sim/object_pose_world.yaml
real2sim_output/debug/real2sim_report.md
```

## 13. 常用清理命令

删除空 episode：

```bash
find demo/fr5_demo/data/il_demos -maxdepth 1 -type d -empty -delete
```

删除运行报告：

```bash
rm -f demo/fr5_demo/data/fr5_live_sync_last_report.json
rm -f demo/fr5_demo/data/fr5_sync_last_report.json
```

删除 Python 缓存：

```bash
find demo/fr5_demo -type d -name __pycache__ -prune -exec rm -rf {} +
```

## 14. 常用故障处理

FR5 SDK 找不到：

```bash
ls fairino-python-sdk-master/linux/fairino/Robot.py
```

如果不存在：

```bash
mkdir -p fairino-python-sdk-master
cp -a ~/fairino-python-sdk-master/linux fairino-python-sdk-master/
```

Astra SDK 找不到：

```bash
ls third_party/pyorbbecsdk/install/lib/pyorbbecsdk*.so
```

如果存在但仍找不到：

```bash
export PYTHONPATH=$PWD/third_party/pyorbbecsdk/install/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/third_party/pyorbbecsdk/install/lib:$LD_LIBRARY_PATH
```

检查当前有效训练数据数量：

```bash
find demo/fr5_demo/data/il_demos -maxdepth 2 -name states.npz | wc -l
```
