# Astra Pro Plus Setup

This project is run through `uv run`, so packages installed with conda `pip install ...`
are not visible unless they are also installed into the project `.venv`.

## OpenCV

OpenCV contrib is tracked in the project dependency file:

```bash
uv sync
uv run python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
uv run python -c "import pip, pybind11; print('pip/pybind11 ok')"
```

## Orbbec SDK

Astra Pro Plus is listed by Orbbec as limited maintenance on `pyorbbecsdk` `main` / v1.x
and not supported on `v2-main`. Use the official `main` branch.

The current PyPI `pyorbbecsdk==1.3.2` wheel is not reliable for this Linux/Python 3.10
environment; in testing it installed a module file that Python could not import.
Build from source instead:

```bash
cd ~/gs_playground
uv sync
mkdir -p third_party
git clone https://github.com/orbbec/pyorbbecsdk.git third_party/pyorbbecsdk
cd third_party/pyorbbecsdk
git checkout main

mkdir -p build
cd build
cmake \
  -DPython3_EXECUTABLE=$(cd ~/gs_playground && uv run python -c 'import sys; print(sys.executable)') \
  -Dpybind11_DIR=$(cd ~/gs_playground && uv run python -m pybind11 --cmakedir) \
  ..
make -j4
make install

cd ~/gs_playground
uv run python -c "import pyorbbecsdk; print('pyorbbecsdk ok')"
```

Do not continue to the `cd`, `cmake`, or `make` commands if `git clone` fails. If
the clone fails, `cd third_party/pyorbbecsdk` will fail and subsequent commands may
run in `~/gs_playground` or `~/gs_playground/build`, which is wrong. The Orbbec
source directory must contain a `CMakeLists.txt`.

Equivalent guarded helper:

```bash
cd ~/gs_playground
bash demo/fr5_demo/setup_orbbec_sdk.sh
```

This helper intentionally stops if `third_party/pyorbbecsdk` is missing, so a
network failure cannot silently run CMake in the wrong directory.

The helper intentionally does not install Orbbec's full `requirements.txt`.
That file pulls optional demo packages such as `open3d`, which is large and not
needed for `camera_capture_orbbec.py`. The project already tracks the packages
we need here: OpenCV contrib, NumPy, pybind11, and pip.

For USB device permission, run the official udev setup once:

```bash
cd ~/gs_playground/third_party/pyorbbecsdk
sudo bash ./scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then retry:

```bash
cd ~/gs_playground
uv run python demo/fr5_demo/camera_capture_orbbec.py
```

For non-interactive capture:

```bash
uv run python demo/fr5_demo/camera_capture_orbbec.py --no-display --max-frames 30 --save-every 1
```

Generate a marker image:

```bash
uv run python demo/fr5_demo/make_aruco_marker.py --id 0 --output data/aruco_marker_0.png
```

For better camera-to-simulation alignment, use two markers on one rigid board:

```bash
uv run python demo/fr5_demo/make_aruco_pair.py --output data/aruco_pair_0_1.png
```

Print without scaling, then measure:

```text
1. black marker side length, excluding white border
2. marker 0 center position in the simulation/world frame
3. marker 1 center position in the simulation/world frame
4. marker board orientation in xyz Euler radians
```

The simulation/world frame used here is the MJCF world frame: table top is
`z=0.000m`, and the FR5 `base_link` mounting surface is `z=0.020m` because of
the added fixed base.

Copy and edit the two-marker template:

```bash
cp demo/fr5_demo/configs/astra_two_markers.template.json \
   demo/fr5_demo/configs/astra_two_markers.json
```

Fill `marker_length_m` and both `marker_world_pos` values. By default the
two-marker config can use:

```json
"auto_marker_world_rpy": true,
"marker_world_normal": [0.0, 0.0, 1.0]
```

With this mode, the solver automatically infers marker orientation from marker 0
center to marker 1 center, projected onto the marker plane. `marker_world_normal`
is the direction the printed marker face points in the MJCF world frame; use
`[0, 0, 1]` for markers lying face-up on the table. The calibration solver uses
all 8 detected corners jointly and prints mean/max reprojection error in pixels.

Capture fresh Astra images:

```bash
uv run python demo/fr5_demo/camera_capture_orbbec.py \
  --no-display \
  --max-frames 60 \
  --save-every 1
```

Then recalibrate using only frames where both markers are visible:

```bash
uv run python demo/fr5_demo/recalibrate_astra.py \
  --marker-spec configs/astra_two_markers.json
```

A good two-marker calibration should normally have low reprojection error. If
the max error is several pixels or more, re-check measured marker length,
marker-center coordinates, board orientation, print scaling, and whether the
paper is flat.

### ChArUco Board Calibration

If alignment is still poor, prefer a ChArUco board over a pure chessboard.
ChArUco uses chessboard corners for subpixel accuracy and ArUco IDs to keep the
board direction unambiguous.

Generate the board and config template:

```bash
uv run python demo/fr5_demo/make_charuco_board.py \
  --output data/charuco_board.png \
  --config-output configs/astra_charuco_board.json
```

Print `data/charuco_board.png` without scaling. Then edit
`configs/astra_charuco_board.json`:

```json
"square_length_m": 0.03,
"marker_length_m": 0.022,
"board_world_origin": [0.0, 0.0, 0.0],
"board_world_x_point": [0.21, 0.0, 0.0],
"board_world_normal": [0.0, 0.0, 1.0]
```

Use the measured printed square size, not the requested printer size. In this
config, `board_world_origin` is the printed ChArUco board's outer top-left
corner in the MJCF world frame. This is the top-left corner of the chessboard
area, excluding any white paper/image margin. It is not the paper corner and not
an ArUco marker corner.

`board_world_x_point` is a measured point along the printed board +X direction.
If you use the outer top-right chessboard corner, then:

```text
distance(board_world_origin, board_world_x_point)
  = squares_x * square_length_m
```

With the default board this is `7 * 0.03m = 0.21m`. For a board lying face-up on
the table, use `board_world_normal: [0, 0, 1]`.

Important frame convention: table top is `z=0.000m`. The FR5 mounting surface
is `z=0.020m` because of the added fixed base. If you measure board height from
the robot mounting surface, subtract `0.020m` before writing MJCF world `z`.

Capture fresh images:

```bash
uv run python demo/fr5_demo/camera_capture_orbbec.py \
  --no-display \
  --max-frames 80 \
  --save-every 1
```

Calibrate from the best captured frame:

```bash
uv run python demo/fr5_demo/calibrate_astra_charuco.py \
  --board-config configs/astra_charuco_board.json
```

The script writes `world_from_camera` into `configs/astra_camera.json`, prints
mean/max reprojection error, and saves an annotated detection image under
`data/astra_detections`. A good result should usually be around subpixel to
about 1 px mean reprojection error; if the max error is several pixels, recheck
print scaling, board flatness, measured board origin, measured board +X point,
and glare/blur in the selected image.

### Dynamic Marker Calibration

If fixed table calibration is hard to measure repeatably, attach a small ArUco
marker to a rigid robot link and let the FR5 move through several small poses.
This is an eye-to-hand calibration: the Astra camera is fixed in the world, and
the marker pose is inferred from the robot link motion.

Generate a 5cm marker:

```bash
uv run python demo/fr5_demo/make_aruco_marker.py \
  --id 0 \
  --output data/fr5_dynamic_marker_0.png
```

Print it without scaling and verify the black marker side is `0.05m`, excluding
the white quiet zone. Tape it rigidly to the moving wrist link. For a marker on
the joint-5 wrist body, start with `--tracking-link wrist2_link`. If the marker
is actually on the tool/flange side after joint 6, use `--tracking-link
wrist3_link`.

Collect samples. The script uses slow `MoveJ`, captures Astra RGB at each pose,
and stores the actual joint readback plus detected marker corners:

```bash
uv run python demo/fr5_demo/fr5_dynamic_marker_calibration.py collect \
  --tracking-link wrist2_link \
  --marker-id 0 \
  --marker-length 0.05 \
  --real-rgb-fps 10 \
  --execute-real
```

It refuses motion if the controller reports an active error code. Clear physical
faults on the teach pendant first; only use `--reset-errors` for resettable
faults.

Solve the camera extrinsic from the latest sample session:

```bash
uv run python demo/fr5_demo/fr5_dynamic_marker_calibration.py solve \
  --tracking-link wrist2_link \
  --marker-id 0 \
  --marker-length 0.05
```

The solver jointly optimizes `world_from_camera` and the fixed marker offset
relative to the selected link, then writes the calibrated camera extrinsic into
`configs/astra_camera.json`. Use at least 8 detected poses, with visible marker
motion in more than one wrist/arm axis. Pure translation or nearly identical
poses are underconstrained.

## Real RGB Overlay

After `configs/astra_camera.json` has valid intrinsics and calibrated extrinsics,
run the FR5 scene from the calibrated Astra camera and overlay the real RGB image:

```bash
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --real-rgb-overlay \
  --real-rgb-alpha 0.45 \
  --real-rgb-widget
```

This attaches a transparent dynamic texture plane in front of the calibrated
`astra_rgb` virtual camera. The main render view is set to `astra_rgb`, so the
real Astra RGB and simulated geometry are seen through the same extrinsics.

If the live camera is busy, the script falls back to the latest saved image in
`data/astra_captures`. To force a static image:

```bash
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --real-rgb-overlay \
  --real-rgb-source latest
```

The overlay above is real RGB plus MuJoCo/MotrixSim geometry. The next section
uses generated FR5 Gaussian assets whose filenames match MotrixSim link names.

## FR5 3DGS Overlay

FR5 mesh-derived Gaussian assets can be generated from the MJCF visual meshes:

```bash
uv run python demo/fr5_demo/generate_fr5_mesh_gaussians.py
```

The generated PLY files are saved in:

```text
demo/fr5_demo/assets/fr5/3dgs_mesh/
```

Run live Astra RGB plus FR5 3DGS pixel overlay:

```bash
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --fr5-gs-overlay
```

Run the live-demo-style view: free 3D camera, moving FR5 simulation, and a
dynamic screen whose texture is rendered from the current simulated FR5 3DGS
camera view:

```bash
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --sim-gs-screen
```

This is the same logic as `demo/live_demo/replay.py`: MotrixSim updates the
robot state, `GSRendererMotrixSim` renders the calibrated camera view from the
current link poses, and the rendered RGB is written into a dynamic texture on a
screen plane in the 3D scene. It is not a live Astra image.

Hide that screen explicitly:

```bash
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --no-sim-gs-screen
```

`--sim-gs-screen` keeps the main view as MotrixSim's free system camera, so you
can move the viewpoint while the robot and the simulator-rendered camera screen
keep updating.

If you have a recorded trajectory, replay it while keeping the same simulated
3DGS screen:

```bash
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --sim-gs-screen \
  --replay-qpos path/to/replay.npz \
  --replay-fps 30
```

`--replay-qpos` accepts `.npz` or `.npy`. For `.npz`, use one of these keys:
`dof_pos`, `qpos`, or `arm_qpos`. A full `dof_pos` frame drives the whole model;
an `arm_qpos` frame with 6 values drives the FR5 arm joints.

Useful variants:

```bash
# Use latest saved Astra RGB for the older real-RGB overlay debug path.
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --fr5-gs-overlay \
  --real-rgb-source latest

# Regenerate FR5 Gaussian assets before launching.
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --fr5-gs-overlay \
  --fr5-gs-regenerate

# Also show the camera-aligned transparent real-RGB plane in the 3D viewport.
uv run python demo/fr5_demo/arm_control.py \
  --camera-config configs/astra_camera.json \
  --real-rgb-overlay \
  --fr5-gs-overlay
```

This is a real `GSRendererMotrixSim` rendering path. The current FR5 assets are
mesh-derived splats, not a photogrammetry-trained 3DGS reconstruction.

For physical/sim synchronized motion, `fr5_live_sync.py` now uses the simulated
3DGS screen by default:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py
```

Disable the screen:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py --no-sim-gs-screen
```

Show raw live Astra only as a small debug widget:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py --real-rgb-widget
```

## Grooved Table

The FR5 MJCF table has been changed from one flat box to a grooved tabletop:

```text
flat 1.5cm + groove 1.0cm + flat 1.5cm = 4.0cm period
```

The table footprint is unchanged from the previous scene:

```text
x: -1.235m to 0.265m
y: -0.760m to 0.240m
top z: 0.000m
```

The groove floor is currently 3mm below the top surface. To change the groove
depth, edit `assets/fr5/mjmodel.xml`: `grooved_table_base` controls the lower
surface and each `grooved_table_flat_*` geom controls the raised 1.5cm strips.

A fixed robot mounting base is also modeled between the table and FR5:

```text
center: same x/y center as the FR5 base frame
size: 0.236m x 0.200m x 0.020m
bottom z: 0.000m
top z: 0.020m
```

The FR5 `base_link` body is lifted to `z=0.020m`, so the robot sits on top of
this fixed base instead of intersecting it.

Quick load check:

```bash
uv run python demo/fr5_demo/arm_control.py --check-only
```

## Sim-First Real Sync

`fr5_sync_sdk.py` is the safe synchronization entrypoint for simulation plus
physical FR5. It always runs a simulation safety check first. By default it does
not send any command to the real robot.

Dry run:

```bash
uv run python demo/fr5_demo/fr5_sync_sdk.py
```

Use a recorded trajectory:

```bash
uv run python demo/fr5_demo/fr5_sync_sdk.py \
  --replay-qpos path/to/replay.npz \
  --source-dt 0.02
```

The safety gates currently check:

```text
joint target inside MJCF actuator limits
max per-frame joint step
minimum TCP z
maximum TCP speed
maximum real ServoJ step after resampling
```

The last dry-run report is written to:

```text
demo/fr5_demo/data/fr5_sync_last_report.json
```

Send the checked trajectory to the physical FR5 only after confirming the robot
is clear and already close to the trajectory start pose:

```bash
uv run python demo/fr5_demo/fr5_sync_sdk.py \
  --robot-ip 192.168.58.2 \
  --execute-real
```

For the USB Robotiq 2F-85 gripper, provide the serial device explicitly:

```bash
uv run python demo/fr5_demo/fr5_sync_sdk.py \
  --robot-ip 192.168.58.2 \
  --execute-real \
  --execute-gripper \
  --gripper-port /dev/ttyUSB0 \
  --gripper-closure 0.0
```

`--gripper-closure` uses `0=open, 1=closed`. The script uses a conservative
Modbus RTU command path for the Robotiq gripper and requires `pyserial` for
actual USB control.

## Live Astra + Sim/Real FR5 Sync

Use `fr5_live_sync.py` when you want the physical FR5 and MotrixSim FR5 to move
from the same joint trajectory while the Astra RGB stream is shown live inside
the simulation scene.

Preflight only:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py --preflight-only
```

Simulation plus live Astra RGB screen, no real robot motion:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py
```

The render window keeps the main view as a free camera. The Astra image appears
as a live screen in the scene and also as a top-left raw RGB widget. The script
now requires a live Astra connection by default; if Astra cannot be opened, it
raises an error instead of silently using a saved image. To explicitly allow a
static fallback:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py --allow-latest-fallback
```

Before moving the real robot, put the physical FR5 at `initial_qpos`:

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py --execute-real
```

Then run synchronized sim + physical motion:

```bash
uv run python demo/fr5_demo/fr5_live_sync.py \
  --execute-real \
  --robot-ip 192.168.58.2
```

The script refuses real motion if the physical joint readback is more than
`--start-tolerance-deg` away from the first trajectory frame. It also runs the
same preflight safety checks before opening the live sync window.

### Cancelling Physical Motion

The physical-motion scripts catch `Ctrl+C` and `SIGTERM`. When cancelled, they
send best-effort stop commands:

```text
ServoMoveEnd, if ServoJ streaming has started
StopMotion
CloseRPC
```

If a process is stuck or you want to cancel from another terminal, run:

```bash
uv run python demo/fr5_demo/fr5_cancel_motion.py
```

If the running script is in ServoJ mode:

```bash
uv run python demo/fr5_demo/fr5_cancel_motion.py --servo-end
```

Only clear resettable controller faults after checking the teach pendant:

```bash
uv run python demo/fr5_demo/fr5_cancel_motion.py --reset-errors
```

## Move Real FR5 To Initial Pose

Use this standalone script before real synchronization when you only need to
move the physical FR5 to `configs/fr5_table_task.json` `initial_qpos`.

Dry run:

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py
```

Actual slow motion:

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py \
  --robot-ip 192.168.58.2 \
  --execute-real
```

Default motion parameters are intentionally slow:

```text
speed_percent=5
MoveJ vel=5
MoveJ acc=5
MoveJ ovl=10
MoveJ blendT=0, non-blocking command plus joint-readback monitoring
```

The script refuses to move if the Fairino SDK cannot read current joint angles,
or if any joint is more than `90deg` away from the initial pose. Use
`--max-delta-deg` to make this stricter.

During real execution the script switches the controller to automatic mode,
enables the robot, sends one non-blocking `MoveJ`, and then monitors the real
joint angles every 0.5s. If the joints do not make measurable progress within
15s or the pose is not reached within 240s, it sends `StopMotion`.

If the SDK reports `-4` / `ERR_RPC_ERROR`, run the safe network diagnosis:

```bash
uv run python demo/fr5_demo/fr5_move_to_initial.py \
  --robot-ip 192.168.58.2 \
  --diagnose-only
```

The Fairino SDK needs controller ports `20003` for XML-RPC commands and `20004`
for realtime state. A direct Ethernet setup usually requires the PC interface to
be on the same subnet, for example `192.168.58.100/24`.

## Imitation Learning Pipeline

Full imitation-learning notes, including Xbox teleop collection and gripper
control, are in:

```text
demo/fr5_demo/README_IMITATION_LEARNING.md
```

The FR5 imitation-learning path is intentionally separated from the sync demos:

```text
real Astra RGB + real SDK joint readback
  -> recorded episode
  -> replay/safety inspection in MotrixSim
  -> behavior cloning training
  -> policy rollout in sim first
  -> optional guarded ServoJ execution on the real FR5
```

### 1. Record Real Demonstrations

The recorder can run in two modes. Without `--xbox-teleop`, it does not command
robot motion and only reads real FR5 joint angles plus Astra RGB. With
`--xbox-teleop`, it prints the gamepad keymap, starts recording only after the
first arm/gripper input, and uses `B` to save the current episode, return to
`initial_qpos`, and wait for the next input.

```bash
uv run python demo/fr5_demo/fr5_record_demonstration.py \
  --robot-ip 192.168.58.2 \
  --hz 10 \
  --duration 20
```

Output is written under:

```text
demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS/
```

Each episode contains:

```text
rgb/*.png
states.npz       # timestamp, joint_deg, joint_rad, next_joint_rad, action_joint_delta_rad, tcp_pos, image_files
replay_qpos.npz  # arm_qpos, directly usable by --replay-qpos
meta.json
```

For open-ended collection, omit `--duration` and stop with `Ctrl+C`.

Xbox segmented collection:

```bash
uv run python demo/fr5_demo/fr5_record_demonstration.py \
  --robot-ip 192.168.58.2 \
  --xbox-teleop \
  --execute-gripper \
  --gripper-port /dev/ttyUSB0 \
  --hz 10
```

Key prompts are printed at startup. `X/Y` close/open the gripper, `A` prints
status, and `B` ends the current episode and returns to the configured initial
pose.

To mix in simulator data, generate episodes with the same schema from a MotrixSim
trajectory and the calibrated Astra virtual camera:

```bash
uv run python demo/fr5_demo/fr5_generate_sim_demonstration.py \
  --hz 10
```

With a saved trajectory:

```bash
uv run python demo/fr5_demo/fr5_generate_sim_demonstration.py \
  --replay-qpos demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS/replay_qpos.npz \
  --source-dt 0.1 \
  --hz 10
```

The generated `sim_episode_*` directory can be trained together with real
episodes because it has the same `states.npz` fields. The RGB source is
simulator-rendered FR5 3DGS, so it is useful for pipeline/debug and sim
augmentation, but it is not a substitute for real demonstrations of contact-rich
manipulation.

### 2. Replay And Inspect

Replay a collected episode in MotrixSim before training or executing policy
rollouts:

```bash
uv run python demo/fr5_demo/fr5_replay_demonstration.py \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS
```

This shows the simulated FR5 and, by default, the recorded RGB frame as a widget.
It can also keep the simulator-rendered 3DGS screen enabled:

```bash
uv run python demo/fr5_demo/fr5_replay_demonstration.py \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS \
  --sim-gs-screen
```

Quick metadata check without a render window:

```bash
uv run python demo/fr5_demo/fr5_replay_demonstration.py \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS \
  --check-only
```

### 3. Train Behavior Cloning

Train a small RGB + joint-state policy to predict the next joint delta:

```bash
uv run python demo/fr5_demo/fr5_train_bc.py \
  --data-dir demo/fr5_demo/data/il_demos \
  --epochs 30 \
  --batch-size 32
```

The default checkpoint is:

```text
demo/fr5_demo/data/il_policies/fr5_bc_last.pt
```

This is a baseline BC model, not a final robust manipulation policy. The first
useful target is to overfit a few short demonstrations and verify that episode
rollout follows the demonstrated joint trajectory in simulation.

### 4. Policy Rollout

Evaluate the trained policy against a recorded episode. This is the first check
after training:

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS
```

For a numeric check without opening the render window:

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --episode demo/fr5_demo/data/il_demos/episode_YYYYmmdd_HHMMSS \
  --no-window
```

Live mode reads the real Astra image and real FR5 joint readback, predicts a
small next joint step, and mirrors that target in simulation only:

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --max-steps 100 \
  --hz 5
```

Physical execution is disabled unless explicitly requested:

```bash
uv run python demo/fr5_demo/fr5_policy_rollout.py \
  --policy demo/fr5_demo/data/il_policies/fr5_bc_last.pt \
  --max-steps 100 \
  --hz 5 \
  --max-action-deg 0.5 \
  --execute-real
```

During real execution the script streams conservative ServoJ targets and checks
the predicted target in simulation before sending each command. If cancelled, it
sends best-effort `ServoMoveEnd()` and closes the RPC connection. For early
experiments keep `--max-action-deg` small and use short `--max-steps`.
