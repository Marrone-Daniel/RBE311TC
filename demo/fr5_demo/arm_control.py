from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DEMO_DIR / "configs" / "fr5_table_task.json"
DEFAULT_CAPTURE_DIR = DEMO_DIR / "data" / "astra_captures"
DEFAULT_FR5_GS_DIR = DEMO_DIR / "assets" / "fr5" / "3dgs_mesh"
OVERLAY_TEXTURE_NAME = "astra_real_tex"
SIM_GS_TEXTURE_NAME = "fr5_sim_gs_tex"

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("ALSOFT_DRIVERS", "null")
os.environ.setdefault("ALSA_CONFIG_PATH", (DEMO_DIR / "configs" / "asound-null.conf").as_posix())
os.environ.setdefault("WINIT_UNIX_BACKEND", "x11")
os.environ.setdefault("WGPU_BACKEND", "vulkan")
os.environ.setdefault("WGPU_POWER_PREF", "high")

import numpy as np
from motrixsim import SceneData, forward_kinematic, msd, step as sim_step
from motrixsim.render import Layout, RenderApp
from camera_geometry import attach_camera_from_config, attach_camera_overlay_from_config, attach_camera_screen_from_config


def resolve_demo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    project_path = DEMO_DIR.parents[1] / path
    if project_path.exists() or path.parts[:2] == ("demo", "fr5_demo"):
        return project_path
    return DEMO_DIR / path


def load_config(path: str | Path) -> dict:
    config_path = resolve_demo_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return config


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for real RGB overlay. Install opencv-python/opencv-contrib-python.") from exc
    return cv2


class RealRgbSource:
    def __init__(
        self,
        *,
        source: str,
        image_path: str | Path | None,
        width: int,
        height: int,
        fps: int,
        allow_fallback: bool = True,
    ):
        self.source = source
        self.image_path = image_path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.allow_fallback = bool(allow_fallback)
        self.cv2 = require_cv2()
        self.pipeline = None
        self.last_rgb: np.ndarray | None = None
        self._frame_to_bgr_image = None

    def start(self) -> np.ndarray:
        if self.source == "live":
            try:
                self._start_live()
                frame = self.read(timeout_ms=1000)
                if frame is None:
                    for _ in range(30):
                        frame = self.read(timeout_ms=100)
                        if frame is not None:
                            break
                if frame is not None:
                    print("Live Astra RGB overlay started.", flush=True)
                    return frame
                self.stop()
                if not self.allow_fallback:
                    raise RuntimeError("Live Astra RGB did not return a frame.")
                print("Warning: live Astra RGB did not return a frame, falling back to latest capture.", flush=True)
            except Exception as exc:
                self.stop()
                if not self.allow_fallback:
                    raise RuntimeError(f"Live Astra RGB failed and fallback is disabled: {exc}") from exc
                print(f"Warning: live Astra RGB overlay failed, falling back to latest capture: {exc}", flush=True)

        frame = self._load_static()
        print("Static Astra RGB overlay loaded.", flush=True)
        return frame

    def _start_live(self) -> None:
        from camera_capture_orbbec import frame_to_bgr_image, require_orbbec

        Config, OBError, OBFormat, OBSensorType, Pipeline, VideoStreamProfile = require_orbbec()
        config = Config()
        pipeline = Pipeline()
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            profile = profiles.get_video_stream_profile(self.width, self.height, OBFormat.RGB, self.fps)
        except OBError:
            profile = profiles.get_default_video_stream_profile()
        config.enable_stream(profile)
        pipeline.start(config)
        self.pipeline = pipeline
        self._frame_to_bgr_image = frame_to_bgr_image

    def _load_static(self) -> np.ndarray:
        path = resolve_demo_path(self.image_path) if self.image_path else self._latest_capture()
        bgr = self.cv2.imread(path.as_posix(), self.cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read Astra RGB overlay image: {path}")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        rgb = self._resize_rgb(rgb)
        self.last_rgb = rgb
        print(f"Astra RGB overlay image: {path}", flush=True)
        return rgb

    def _latest_capture(self) -> Path:
        captures = sorted(DEFAULT_CAPTURE_DIR.glob("astra_rgb_*.png"))
        if not captures:
            raise RuntimeError(f"No Astra capture images found in {DEFAULT_CAPTURE_DIR}")
        return captures[-1]

    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.shape[:2] == (self.height, self.width):
            return np.ascontiguousarray(rgb)
        resized = self.cv2.resize(rgb, (self.width, self.height), interpolation=self.cv2.INTER_LINEAR)
        return np.ascontiguousarray(resized)

    def read(self, timeout_ms: int = 5) -> np.ndarray | None:
        if self.pipeline is None:
            return self.last_rgb
        frames = self.pipeline.wait_for_frames(int(timeout_ms))
        if frames is None:
            return self.last_rgb
        return self._frames_to_rgb(frames)

    def read_latest(self, *, timeout_ms: int = 1, max_drain: int = 12) -> np.ndarray | None:
        if self.pipeline is None:
            return self.last_rgb
        latest = None
        frames = self.pipeline.wait_for_frames(int(timeout_ms))
        if frames is not None:
            latest = frames
        for _ in range(max(0, int(max_drain))):
            frames = self.pipeline.wait_for_frames(1)
            if frames is None:
                break
            latest = frames
        if latest is None:
            return self.last_rgb
        return self._frames_to_rgb(latest)

    def _frames_to_rgb(self, frames) -> np.ndarray | None:
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return self.last_rgb
        bgr = self._frame_to_bgr_image(color_frame)
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        rgb = self._resize_rgb(rgb)
        self.last_rgb = rgb
        return rgb

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


def require_names(kind: str, required: list[str], available: list[str]) -> None:
    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(f"Missing {kind}: {missing}. Available {kind}: {available}")


def joint_qpos_indices(model, joint_names: list[str]) -> np.ndarray:
    indices = []
    for name in joint_names:
        joint_id = model.get_joint_index(name)
        dof_num = int(model.joint_dof_pos_nums[joint_id])
        if dof_num != 1:
            raise RuntimeError(f"Expected scalar joint {name!r}, got dof_pos count {dof_num}")
        indices.append(int(model.joint_dof_pos_indices[joint_id]))
    return np.asarray(indices, dtype=np.int64)


def actuator_indices(model, actuator_names: list[str]) -> np.ndarray:
    return np.asarray([model.get_actuator_index(name) for name in actuator_names], dtype=np.int64)


def clip_actuator_targets(model, act_ids: np.ndarray, targets: np.ndarray) -> np.ndarray:
    limits = np.asarray(model.actuator_ctrl_limits, dtype=np.float32)
    lows = limits[0, act_ids]
    highs = limits[1, act_ids]
    return np.clip(np.asarray(targets, dtype=np.float32), lows, highs)


def write_body_ctrls(data: SceneData, body, ctrl: np.ndarray) -> None:
    body.set_actuator_ctrls(data, np.asarray(ctrl, dtype=np.float32))


def set_arm_qpos_and_ctrl(data: SceneData, model, body, qpos_ids: np.ndarray, act_ids: np.ndarray, q: np.ndarray) -> None:
    q = clip_actuator_targets(model, act_ids, q)
    full_qpos = np.asarray(data.dof_pos, dtype=np.float32).copy()
    full_qpos[qpos_ids] = q
    data.set_dof_pos(full_qpos, model)
    data.set_dof_vel(np.zeros_like(data.dof_vel))
    ctrl = np.asarray(data.actuator_ctrls, dtype=np.float32).copy()
    ctrl[act_ids] = q
    write_body_ctrls(data, body, ctrl)


def set_arm_ctrl(data: SceneData, model, body, act_ids: np.ndarray, q: np.ndarray) -> None:
    q = clip_actuator_targets(model, act_ids, q)
    ctrl = np.asarray(data.actuator_ctrls, dtype=np.float32).copy()
    ctrl[act_ids] = q
    write_body_ctrls(data, body, ctrl)


def set_gripper(data: SceneData, model, body, gripper_act_ids: np.ndarray, opening: float) -> None:
    if gripper_act_ids.shape[0] == 0:
        return
    opening = float(np.clip(opening, 0.0, 1.0))
    if gripper_act_ids.shape[0] == 1:
        targets = np.asarray([0.82 * opening], dtype=np.float32)
    elif gripper_act_ids.shape[0] == 2:
        targets = np.asarray([0.8 * opening, -0.8 * opening], dtype=np.float32)
    else:
        raise RuntimeError(f"Expected 1 or 2 gripper actuators, got {gripper_act_ids.shape[0]}")
    targets = clip_actuator_targets(model, gripper_act_ids, targets)
    ctrl = np.asarray(data.actuator_ctrls, dtype=np.float32).copy()
    ctrl[gripper_act_ids] = targets
    write_body_ctrls(data, body, ctrl)


def site_position(model, data: SceneData, site_name: str) -> np.ndarray:
    forward_kinematic(model, data)
    site = model.get_site(site_name)
    pose = np.asarray(site.get_pose(data), dtype=np.float32)
    return pose[:3].copy()


def print_model_summary(model, config: dict) -> None:
    print("FR5 MotrixSim model loaded.")
    print(f"  links({len(model.link_names)}): {model.link_names}")
    print(f"  joints({len(model.joint_names)}): {model.joint_names}")
    print(f"  actuators({len(model.actuator_names)}): {model.actuator_names}")
    print(f"  sites({len(model.site_names)}): {model.site_names}")
    camera_names = [camera.name for camera in model.cameras]
    print(f"  cameras({len(camera_names)}): {camera_names}")
    print(f"  timestep: {float(model.options.timestep):.6f}s")
    print(f"  robot: {config.get('robot', 'fr5')}")


def camera_name_from_config(camera_config_path: str | Path | None) -> str | None:
    if not camera_config_path:
        return None
    cfg_path = resolve_demo_path(camera_config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return str(cfg.get("name", "") or "") or None


def find_camera(model, name: str | None):
    if name:
        for camera in model.cameras:
            if camera.name == name or camera.name.endswith(name):
                return camera
    if len(model.cameras) > 0:
        return model.cameras[0]
    return None


def find_camera_id(model, name: str | None) -> int | None:
    if name:
        for idx, camera in enumerate(model.cameras):
            if camera.name == name or camera.name.endswith(name):
                return int(idx)
    if len(model.cameras) > 0:
        return 0
    return None


def fr5_gs_settings(config: dict, *, points_per_geom=None, scale=None, opacity=None) -> tuple[int, float, float]:
    gs_cfg = config.get("fr5_3dgs", {}) if isinstance(config.get("fr5_3dgs", {}), dict) else {}
    p = int(points_per_geom if points_per_geom is not None else gs_cfg.get("points_per_geom", 10000))
    s = float(scale if scale is not None else gs_cfg.get("scale", 0.0011))
    o = float(opacity if opacity is not None else gs_cfg.get("opacity", 0.58))
    return p, s, o


def ensure_fr5_gaussian_assets(
    config: dict,
    gs_dir: Path,
    *,
    regenerate: bool,
    points_per_geom: int | None = None,
    scale: float | None = None,
    opacity: float | None = None,
) -> None:
    manifest = gs_dir / "manifest.json"
    if manifest.exists() and not regenerate:
        return
    from generate_fr5_mesh_gaussians import generate_fr5_mesh_gaussians

    model_xml = resolve_demo_path(config["model_xml"])
    p, s, o = fr5_gs_settings(config, points_per_geom=points_per_geom, scale=scale, opacity=opacity)
    print(f"Generating FR5 mesh-derived Gaussian assets in {gs_dir}...", flush=True)
    generate_fr5_mesh_gaussians(
        model_xml,
        gs_dir,
        points_per_geom=p,
        scale=s,
        opacity=o,
    )


def collect_fr5_gaussian_assets(model, gs_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for link_name in model.link_names:
        ply = gs_dir / f"{link_name}.ply"
        if ply.exists():
            out[link_name] = ply.as_posix()
    return out


def camera_resolution_from_config(camera_config_path: str | Path) -> tuple[int, int]:
    cfg_path = resolve_demo_path(camera_config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    intr = cfg.get("intrinsics", {})
    return int(intr.get("width", 640)), int(intr.get("height", 480))


def tensor_image_to_u8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def tensor_depth_mask(depth, rgb: np.ndarray) -> np.ndarray:
    if depth is not None and hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    if depth is not None:
        d = np.asarray(depth)
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        if d.ndim == 2:
            mask = np.isfinite(d) & (d > 0.0)
            if mask.any():
                return mask
    return rgb.max(axis=-1) > 3


def blend_gs_over_rgb(real_rgb: np.ndarray, gs_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if real_rgb.shape[:2] != gs_rgb.shape[:2]:
        cv2 = require_cv2()
        real_rgb = cv2.resize(real_rgb, (gs_rgb.shape[1], gs_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = real_rgb.astype(np.float32).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    out[mask] = (1.0 - a) * out[mask] + a * gs_rgb.astype(np.float32)[mask]
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def render_fr5_gs_rgb(gs_renderer, model, data, camera_id: int, width: int, height: int, system_camera=None) -> np.ndarray:
    forward_kinematic(model, data)
    gs_renderer.update_gaussians(data)
    results = gs_renderer.render(
        model,
        data,
        [int(camera_id)],
        int(width),
        int(height),
        system_camera=system_camera,
    )
    gs_rgb_t, _ = results[int(camera_id)]
    return tensor_image_to_u8(gs_rgb_t)


def load_replay_qpos(path: str | Path) -> np.ndarray:
    replay_path = resolve_demo_path(path)
    if replay_path.suffix == ".npy":
        arr = np.load(replay_path.as_posix())
    else:
        pack = np.load(replay_path.as_posix())
        for key in ("dof_pos", "qpos", "arm_qpos"):
            if key in pack:
                arr = pack[key]
                break
        else:
            raise RuntimeError(f"{replay_path} must contain one of: dof_pos, qpos, arm_qpos")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Replay qpos must be 2D, got shape {arr.shape}")
    print(f"Loaded replay qpos: {replay_path}, shape={arr.shape}", flush=True)
    return arr


def build_runtime(
    config: dict,
    camera_config_path: str | Path | None = None,
    *,
    real_rgb_overlay: bool = False,
    real_rgb_screen: bool = False,
    sim_gs_screen: bool = False,
    overlay_distance: float = 0.75,
    overlay_alpha: float = 0.45,
):
    model_xml = resolve_demo_path(config["model_xml"])
    scene = msd.from_file(model_xml.as_posix())
    if camera_config_path:
        camera_config = resolve_demo_path(camera_config_path)
        attach_camera_from_config(scene, camera_config)
        if real_rgb_overlay:
            attach_camera_overlay_from_config(scene, camera_config, distance=overlay_distance, alpha=overlay_alpha)
        elif real_rgb_screen:
            attach_camera_screen_from_config(scene, camera_config, distance=overlay_distance, alpha=overlay_alpha)
        elif sim_gs_screen:
            attach_camera_screen_from_config(
                scene,
                camera_config,
                texture_name=SIM_GS_TEXTURE_NAME,
                material_name="fr5_sim_gs_mat",
                edge_material_name="fr5_sim_gs_edge_mat",
                distance=overlay_distance,
                alpha=overlay_alpha,
            )
    model = scene.build()

    joint_names = list(config["joint_names"])
    actuator_names = list(config["actuator_names"])
    gripper_names = list(config.get("gripper_actuator_names", []))
    tcp_site = str(config["tcp_site"])

    require_names("joints", joint_names, model.joint_names)
    require_names("actuators", actuator_names, model.actuator_names)
    require_names("gripper actuators", gripper_names, model.actuator_names)
    require_names("sites", [tcp_site], model.site_names)

    qpos_ids = joint_qpos_indices(model, joint_names)
    arm_act_ids = actuator_indices(model, actuator_names)
    gripper_act_ids = actuator_indices(model, gripper_names) if gripper_names else np.asarray([], dtype=np.int64)
    body = model.get_body("base_link")

    data = SceneData(model)
    initial_qpos = np.asarray(config["initial_qpos"], dtype=np.float32)
    if initial_qpos.shape != (len(joint_names),):
        raise ValueError(f"initial_qpos must have {len(joint_names)} values")

    set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, initial_qpos)
    set_gripper(data, model, body, gripper_act_ids, float(config.get("gripper_opening", 0.0)))
    forward_kinematic(model, data)
    return model, data, body, qpos_ids, arm_act_ids, gripper_act_ids


def run_check(
    config: dict,
    camera_config_path: str | Path | None = None,
    *,
    real_rgb_overlay: bool = False,
    real_rgb_screen: bool = False,
    sim_gs_screen: bool = False,
) -> None:
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(
        config,
        camera_config_path,
        real_rgb_overlay=real_rgb_overlay,
        real_rgb_screen=real_rgb_screen,
        sim_gs_screen=sim_gs_screen,
    )
    print_model_summary(model, config)
    print(f"  arm qpos indices: {qpos_ids.tolist()}")
    print(f"  arm actuator indices: {arm_act_ids.tolist()}")
    print(f"  gripper actuator indices: {gripper_act_ids.tolist()}")
    print(f"  initial arm qpos: {data.dof_pos[qpos_ids].tolist()}")
    print(f"  initial actuator ctrl: {data.actuator_ctrls[arm_act_ids].tolist()}")
    print(f"  tcp position: {site_position(model, data, config['tcp_site']).tolist()}")
    cube_site = config.get("cube_site")
    if cube_site and cube_site in model.site_names:
        print(f"  cube position: {site_position(model, data, cube_site).tolist()}")

    target_qpos = np.asarray(config.get("demo_target_qpos", config["initial_qpos"]), dtype=np.float32)
    set_arm_ctrl(data, model, body, arm_act_ids, target_qpos)
    before = np.asarray(data.dof_pos[qpos_ids], dtype=np.float32).copy()
    for _ in range(100):
        sim_step(model, data)
    after = np.asarray(data.dof_pos[qpos_ids], dtype=np.float32).copy()
    delta = after - before
    print(f"  post-step arm qpos: {after.tolist()}")
    print(f"  post-step delta: {delta.tolist()}")


def run_window(config: dict, camera_config_path: str | Path | None = None, args=None) -> None:
    real_rgb_overlay = bool(args and args.real_rgb_overlay)
    real_rgb_screen = bool(args and args.real_rgb_screen)
    fr5_gs_overlay = bool(args and args.fr5_gs_overlay)
    sim_gs_screen = bool(args and args.sim_gs_screen)
    model, data, body, qpos_ids, arm_act_ids, gripper_act_ids = build_runtime(
        config,
        camera_config_path,
        real_rgb_overlay=real_rgb_overlay,
        real_rgb_screen=real_rgb_screen,
        sim_gs_screen=sim_gs_screen,
        overlay_distance=float(args.real_rgb_distance) if args else 0.75,
        overlay_alpha=float(args.sim_gs_screen_alpha if sim_gs_screen else args.real_rgb_alpha if not real_rgb_screen else args.real_rgb_screen_alpha) if args else 0.45,
    )
    print_model_summary(model, config)
    attached_camera_name = camera_name_from_config(camera_config_path)
    attached_camera_id = find_camera_id(model, attached_camera_name)

    target_qpos = np.asarray(config.get("demo_target_qpos", config["initial_qpos"]), dtype=np.float32)
    if target_qpos.shape != (len(config["joint_names"]),):
        raise ValueError(f"demo_target_qpos must have {len(config['joint_names'])} values")
    replay_qpos = load_replay_qpos(args.replay_qpos) if args and args.replay_qpos else None

    control_dt = float(config.get("control_dt", 0.02))
    if replay_qpos is not None:
        control_dt = 1.0 / max(1.0, float(args.replay_fps))
    steps_per_ctrl = max(1, round(control_dt / float(model.options.timestep)))
    tick = 0
    rgb_source = None
    gs_renderer = None
    gs_width = int(args.fr5_gs_width)
    gs_height = int(args.fr5_gs_height)
    if sim_gs_screen and camera_config_path:
        gs_width, gs_height = camera_resolution_from_config(camera_config_path)
    needs_fr5_gs = fr5_gs_overlay or sim_gs_screen
    if needs_fr5_gs:
        if not camera_config_path:
            raise RuntimeError("--fr5-gs-overlay/--sim-gs-screen requires --camera-config configs/astra_camera.json")
        if attached_camera_id is None:
            raise RuntimeError("No calibrated model camera available for FR5 3DGS rendering.")
        gs_dir = resolve_demo_path(args.fr5_gs_dir)
        ensure_fr5_gaussian_assets(
            config,
            gs_dir,
            regenerate=bool(args.fr5_gs_regenerate),
            points_per_geom=args.fr5_gs_points_per_geom,
        )
        gaussians = collect_fr5_gaussian_assets(model, gs_dir)
        if not gaussians:
            raise RuntimeError(f"No FR5 Gaussian PLY assets found in {gs_dir}")
        print(f"Loading FR5 Gaussian assets({len(gaussians)}): {', '.join(sorted(gaussians))}", flush=True)
        from gaussian_renderer import GSRendererMotrixSim

        gs_renderer = GSRendererMotrixSim(gaussians, model)
        print("FR5 3DGS renderer ready.", flush=True)

    print("Constructing RenderApp...", flush=True)
    with RenderApp() as render:
        print("Launching render window...", flush=True)
        render.launch(model)
        lock_main_to_camera = bool(
            camera_config_path
            and (
                bool(args.lock_view_to_camera)
                or (not args.free_view and not real_rgb_screen and not sim_gs_screen)
            )
        )
        if lock_main_to_camera:
            camera = find_camera(model, attached_camera_name)
            if camera is None:
                print("Warning: camera config was provided, but no model camera is available.", flush=True)
            else:
                render.set_main_camera(camera)
                print(f"Main render view set to camera: {camera.name}", flush=True)
        elif camera_config_path:
            print("Main render view left as free system camera.", flush=True)
        print(f"Render launch returned. is_closed={render.is_closed}", flush=True)

        overlay_texture = None
        sim_gs_texture = None
        overlay_widget_img = None
        gs_overlay_widget_img = None
        sim_gs_widget_img = None
        needs_real_rgb = real_rgb_overlay or real_rgb_screen or fr5_gs_overlay
        if needs_real_rgb:
            rgb_source = RealRgbSource(
                source=str(args.real_rgb_source),
                image_path=args.real_rgb_image or None,
                width=int(args.real_rgb_width if not fr5_gs_overlay else gs_width),
                height=int(args.real_rgb_height if not fr5_gs_overlay else gs_height),
                fps=int(args.real_rgb_fps),
            )
            first_rgb = rgb_source.start()
            if fr5_gs_overlay and args.fr5_gs_widget:
                gs_overlay_widget_img = render.create_image(first_rgb)
                render.widgets.create_image_widget(
                    gs_overlay_widget_img,
                    layout=Layout(
                        left=int(args.fr5_gs_widget_left),
                        top=int(args.fr5_gs_widget_top),
                        width=int(args.fr5_gs_widget_width),
                        height=int(args.fr5_gs_widget_height),
                    ),
                )
        if real_rgb_overlay or real_rgb_screen:
            overlay_texture = render.get_texture_image(OVERLAY_TEXTURE_NAME)
            overlay_texture.pixels = np.ascontiguousarray(first_rgb)
            if args.real_rgb_widget:
                overlay_widget_img = render.create_image(first_rgb)
                render.widgets.create_image_widget(
                    overlay_widget_img,
                    layout=Layout(
                        left=10,
                        top=10,
                        width=int(args.real_rgb_widget_width),
                        height=int(args.real_rgb_widget_height),
                    ),
                )
        if sim_gs_screen:
            if gs_renderer is None or attached_camera_id is None:
                raise RuntimeError("--sim-gs-screen requires a ready FR5 3DGS renderer and calibrated camera")
            sim_rgb = render_fr5_gs_rgb(
                gs_renderer,
                model,
                data,
                int(attached_camera_id),
                gs_width,
                gs_height,
                system_camera=render.system_camera,
            )
            sim_gs_texture = render.get_texture_image(SIM_GS_TEXTURE_NAME)
            sim_gs_texture.pixels = np.ascontiguousarray(sim_rgb)
            if args.sim_gs_widget:
                sim_gs_widget_img = render.create_image(sim_rgb)
                render.widgets.create_image_widget(
                    sim_gs_widget_img,
                    layout=Layout(
                        left=int(args.sim_gs_widget_left),
                        top=int(args.sim_gs_widget_top),
                        width=int(args.sim_gs_widget_width),
                        height=int(args.sim_gs_widget_height),
                    ),
                )

        render.sync(data)
        print(f"Render sync completed. is_closed={render.is_closed}", flush=True)
        print("Render window launched. Close the window to exit.", flush=True)

        try:
            rgb_update_period = max(1, round(1.0 / (control_dt * max(1, int(args.real_rgb_fps)))) if args else 1)
            sim_gs_update_period = max(1, round(1.0 / (control_dt * max(1, int(args.sim_gs_fps)))) if args else 1)
            while not render.is_closed:
                if replay_qpos is not None:
                    q_frame = replay_qpos[tick % replay_qpos.shape[0]]
                    if q_frame.shape[0] == data.dof_pos.shape[0]:
                        data.set_dof_pos(q_frame, model)
                        data.set_dof_vel(np.zeros_like(data.dof_vel))
                    elif q_frame.shape[0] == len(qpos_ids):
                        set_arm_qpos_and_ctrl(data, model, body, qpos_ids, arm_act_ids, q_frame)
                    else:
                        raise RuntimeError(
                            f"Replay frame has {q_frame.shape[0]} qpos values; expected {data.dof_pos.shape[0]} full dof_pos or {len(qpos_ids)} arm qpos"
                        )
                else:
                    alpha = 0.5 + 0.5 * np.sin(tick * control_dt)
                    q = (1.0 - alpha) * np.asarray(config["initial_qpos"], dtype=np.float32) + alpha * target_qpos
                    set_arm_ctrl(data, model, body, arm_act_ids, q)
                set_gripper(data, model, body, gripper_act_ids, float(config.get("gripper_opening", 0.0)))

                for _ in range(steps_per_ctrl):
                    sim_step(model, data)

                if sim_gs_screen and sim_gs_texture is not None and gs_renderer is not None and attached_camera_id is not None and tick % sim_gs_update_period == 0:
                    sim_rgb = render_fr5_gs_rgb(
                        gs_renderer,
                        model,
                        data,
                        int(attached_camera_id),
                        gs_width,
                        gs_height,
                        system_camera=render.system_camera,
                    )
                    sim_gs_texture.pixels = np.ascontiguousarray(sim_rgb)
                    if sim_gs_widget_img is not None:
                        sim_gs_widget_img.pixels = sim_rgb

                if rgb_source is not None and tick % rgb_update_period == 0:
                    rgb = rgb_source.read()
                    if rgb is not None:
                        rgb = np.ascontiguousarray(rgb)
                        if overlay_widget_img is not None:
                            overlay_widget_img.pixels = rgb
                        screen_rgb = rgb
                        if fr5_gs_overlay and gs_renderer is not None and (gs_overlay_widget_img is not None or real_rgb_screen):
                            forward_kinematic(model, data)
                            gs_renderer.update_gaussians(data)
                            results = gs_renderer.render(
                                model,
                                data,
                                [int(attached_camera_id)],
                                gs_width,
                                gs_height,
                                system_camera=render.system_camera,
                            )
                            gs_rgb_t, gs_depth_t = results[int(attached_camera_id)]
                            gs_rgb = tensor_image_to_u8(gs_rgb_t)
                            mask = tensor_depth_mask(gs_depth_t, gs_rgb)
                            blended = blend_gs_over_rgb(rgb, gs_rgb, mask, float(args.fr5_gs_alpha))
                            if gs_overlay_widget_img is not None:
                                gs_overlay_widget_img.pixels = blended
                            if real_rgb_screen:
                                screen_rgb = blended
                        if overlay_texture is not None:
                            overlay_texture.pixels = np.ascontiguousarray(screen_rgb)

                if tick % 50 == 0:
                    tcp = site_position(model, data, config["tcp_site"])
                    print(f"tick={tick:05d} arm_qpos={data.dof_pos[qpos_ids].round(4).tolist()} tcp={tcp.round(4).tolist()}")

                render.sync(data)
                tick += 1
                if args and int(args.max_ticks) > 0 and tick >= int(args.max_ticks):
                    break
                time.sleep(control_dt)
        finally:
            if rgb_source is not None:
                rgb_source.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="FR5 first-stage MotrixSim load and joint-control demo")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--check-only", action="store_true", help="Validate model/config without opening RenderApp")
    parser.add_argument("--replay-qpos", type=str, default="", help="Optional .npz/.npy replay trajectory with dof_pos/qpos/arm_qpos")
    parser.add_argument("--replay-fps", type=float, default=30.0)
    parser.add_argument("--max-ticks", type=int, default=0, help="Stop after N control ticks; 0 means run until the window closes")
    parser.add_argument(
        "--render-backend",
        choices=["x11", "wayland", "auto"],
        default=os.environ.get("WINIT_UNIX_BACKEND", "x11"),
        help="Window backend for RenderApp. x11 is usually more stable on NVIDIA hybrid laptops.",
    )
    parser.add_argument("--force-nvidia", action="store_true", help="Force NVIDIA PRIME offload for RenderApp")
    parser.add_argument(
        "--camera-config",
        type=str,
        default="",
        help="Attach a calibrated virtual camera from this JSON config, e.g. configs/astra_camera.json",
    )
    parser.add_argument("--real-rgb-overlay", action="store_true", help="Overlay Astra RGB as a transparent camera-aligned texture plane")
    parser.add_argument("--real-rgb-screen", action="store_true", help="Show Astra RGB on a live_demo-style dynamic screen in the 3D scene")
    parser.add_argument("--real-rgb-screen-alpha", type=float, default=1.0, help="Opacity for the live RGB screen material")
    parser.add_argument(
        "--sim-gs-screen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show a live_demo-style screen whose texture is rendered from simulated FR5 3DGS, not real Astra RGB",
    )
    parser.add_argument("--sim-gs-screen-alpha", type=float, default=1.0, help="Opacity for the simulated 3DGS screen material")
    parser.add_argument("--sim-gs-fps", type=int, default=15, help="Update rate for the simulator-rendered 3DGS screen")
    parser.add_argument("--sim-gs-widget", action=argparse.BooleanOptionalAction, default=False, help="Also show the simulated 3DGS frame as a UI widget")
    parser.add_argument("--sim-gs-widget-left", type=int, default=10)
    parser.add_argument("--sim-gs-widget-top", type=int, default=10)
    parser.add_argument("--sim-gs-widget-width", type=int, default=640)
    parser.add_argument("--sim-gs-widget-height", type=int, default=480)
    parser.add_argument("--free-view", action="store_true", help="Keep the main window on the free system camera instead of locking to astra_rgb")
    parser.add_argument(
        "--lock-view-to-camera",
        action="store_true",
        help="Force the main render view to the calibrated astra_rgb camera, including screen/overlay modes.",
    )
    parser.add_argument("--real-rgb-source", choices=["live", "latest"], default="live", help="Live Astra stream, or latest saved Astra capture")
    parser.add_argument("--real-rgb-image", type=str, default="", help="Specific RGB image for static overlay")
    parser.add_argument("--real-rgb-alpha", type=float, default=0.45, help="Overlay opacity, 0 transparent to 1 opaque")
    parser.add_argument("--real-rgb-distance", type=float, default=0.75, help="Distance in front of calibrated camera for overlay plane")
    parser.add_argument("--real-rgb-width", type=int, default=640)
    parser.add_argument("--real-rgb-height", type=int, default=480)
    parser.add_argument("--real-rgb-fps", type=int, default=15)
    parser.add_argument("--real-rgb-widget", action="store_true", help="Also show raw Astra RGB in a small top-left widget")
    parser.add_argument("--real-rgb-widget-width", type=int, default=320)
    parser.add_argument("--real-rgb-widget-height", type=int, default=240)
    parser.add_argument("--fr5-gs-overlay", action="store_true", help="Render FR5 Gaussian splats from the calibrated Astra camera and blend them over real RGB")
    parser.add_argument("--fr5-gs-dir", type=str, default=DEFAULT_FR5_GS_DIR.as_posix())
    parser.add_argument("--fr5-gs-regenerate", action="store_true", help="Regenerate mesh-derived FR5 Gaussian PLY assets before launching")
    parser.add_argument("--fr5-gs-points-per-geom", type=int, default=None, help="Override config fr5_3dgs.points_per_geom")
    parser.add_argument("--fr5-gs-alpha", type=float, default=0.75, help="FR5 3DGS opacity in the blended overlay widget")
    parser.add_argument("--fr5-gs-width", type=int, default=640)
    parser.add_argument("--fr5-gs-height", type=int, default=480)
    parser.add_argument("--fr5-gs-widget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fr5-gs-widget-left", type=int, default=10)
    parser.add_argument("--fr5-gs-widget-top", type=int, default=10)
    parser.add_argument("--fr5-gs-widget-width", type=int, default=640)
    parser.add_argument("--fr5-gs-widget-height", type=int, default=480)
    args = parser.parse_args()

    if args.render_backend == "auto":
        os.environ.pop("WINIT_UNIX_BACKEND", None)
    else:
        os.environ["WINIT_UNIX_BACKEND"] = args.render_backend
    if args.force_nvidia:
        os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    else:
        os.environ.pop("__NV_PRIME_RENDER_OFFLOAD", None)
        os.environ.pop("__GLX_VENDOR_LIBRARY_NAME", None)

    if args.real_rgb_overlay and not args.camera_config:
        raise RuntimeError("--real-rgb-overlay requires --camera-config configs/astra_camera.json")
    if args.real_rgb_screen and not args.camera_config:
        raise RuntimeError("--real-rgb-screen requires --camera-config configs/astra_camera.json")
    if args.real_rgb_overlay and args.real_rgb_screen:
        raise RuntimeError("Use only one of --real-rgb-overlay or --real-rgb-screen; they share the same dynamic texture.")
    if args.fr5_gs_overlay and not args.camera_config:
        raise RuntimeError("--fr5-gs-overlay requires --camera-config configs/astra_camera.json")
    if args.sim_gs_screen and not args.camera_config:
        raise RuntimeError("--sim-gs-screen requires --camera-config configs/astra_camera.json")
    if args.sim_gs_screen and (args.real_rgb_overlay or args.real_rgb_screen):
        raise RuntimeError("--sim-gs-screen is the simulator-rendered screen. Do not combine it with --real-rgb-overlay/--real-rgb-screen.")

    config = load_config(args.config)
    if args.check_only:
        run_check(
            config,
            args.camera_config or None,
            real_rgb_overlay=bool(args.real_rgb_overlay),
            real_rgb_screen=bool(args.real_rgb_screen),
            sim_gs_screen=bool(args.sim_gs_screen),
        )
    else:
        run_window(config, args.camera_config or None, args=args)


if __name__ == "__main__":
    main()
