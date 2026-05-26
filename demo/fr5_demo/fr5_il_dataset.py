from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_IL_DEMO_DIR = DEMO_DIR / "data" / "il_demos"
DEFAULT_POLICY_DIR = DEMO_DIR / "data" / "il_policies"
ROBOT_DOF = 6
STATE_DIM = 7
TARGET_FEATURE_DIM = 6
DEFAULT_MODEL_TYPE = "cnn_small"
DEFAULT_STATE_CLIP = 5.0
DEFAULT_TARGET_FEATURE_STD_FLOOR = 0.05
DEFAULT_ACTION_SCALE_DEG = 18.0
DEFAULT_GRIPPER_ACTION_SCALE = 0.1


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


def load_episode_npz(episode_dir: str | Path) -> dict[str, np.ndarray]:
    episode_dir = resolve_demo_path(episode_dir)
    states_path = episode_dir / "states.npz"
    if not states_path.exists():
        raise FileNotFoundError(f"Missing episode state file: {states_path}")
    pack = np.load(states_path.as_posix(), allow_pickle=True)
    required = ("joint_rad", "action_joint_delta_rad", "image_files")
    missing = [key for key in required if key not in pack]
    if missing:
        raise RuntimeError(f"{states_path} missing keys: {missing}")
    return {key: pack[key] for key in pack.files}


def load_episode_meta(episode_dir: str | Path) -> dict:
    episode_dir = resolve_demo_path(episode_dir)
    meta_path = episode_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_episode_dirs(data_dir: str | Path, episodes: Iterable[str] | None = None) -> list[Path]:
    data_dir = resolve_demo_path(data_dir)
    if episodes:
        out = []
        for item in episodes:
            path = resolve_demo_path(item)
            if not path.exists():
                path = data_dir / item
            out.append(path.resolve())
    else:
        out = sorted(path for path in data_dir.iterdir() if (path / "states.npz").exists()) if data_dir.exists() else []
    if not out:
        raise RuntimeError(f"No imitation-learning episodes found under {data_dir}")
    return out


@dataclass(frozen=True)
class BcSample:
    episode_dir: Path
    image_file: str
    joint_rad: np.ndarray
    gripper_closure: float
    target_feature: np.ndarray
    action_delta_rad: np.ndarray
    action_gripper_delta: float
    reward: float


def camera_projection_from_config(camera_config: str | Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    cfg_path = resolve_demo_path(camera_config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    intr = cfg.get("intrinsics", {})
    ext = cfg.get("extrinsics", {})
    k = np.asarray(
        [
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t_world_camera = np.eye(4, dtype=np.float64)
    t_world_camera[:3, :3] = np.asarray(ext["rotation_matrix"], dtype=np.float64)
    t_world_camera[:3, 3] = np.asarray(ext["position"], dtype=np.float64)
    return k, np.linalg.inv(t_world_camera), int(intr["width"]), int(intr["height"])


def project_point_feature(point_world: np.ndarray, k: np.ndarray, t_camera_world: np.ndarray, width: int, height: int) -> np.ndarray:
    point = np.asarray(point_world, dtype=np.float64).reshape(3)
    homog = np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    cam = (t_camera_world @ homog)[:3]
    z = float(cam[2])
    if z <= 1e-6:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    u = float(k[0, 0] * cam[0] / z + k[0, 2])
    v = float(k[1, 1] * cam[1] / z + k[1, 2])
    return np.asarray(
        [
            (u / max(1.0, float(width) - 1.0)) * 2.0 - 1.0,
            (v / max(1.0, float(height) - 1.0)) * 2.0 - 1.0,
            1.0 / max(1e-6, z),
        ],
        dtype=np.float32,
    )


def make_target_feature(
    object_pos_m: np.ndarray,
    goal_pos_m: np.ndarray,
    *,
    camera_config: str | Path,
) -> np.ndarray:
    k, t_camera_world, width, height = camera_projection_from_config(camera_config)
    return np.concatenate(
        [
            project_point_feature(object_pos_m, k, t_camera_world, width, height),
            project_point_feature(goal_pos_m, k, t_camera_world, width, height),
        ],
        axis=0,
    ).astype(np.float32)


def target_feature_from_episode(episode_dir: Path, *, target_conditioning: bool) -> np.ndarray:
    if not target_conditioning:
        return np.zeros(0, dtype=np.float32)
    report_path = episode_dir / "pick_place_report.json"
    meta_path = episode_dir / "meta.json"
    source_path = report_path if report_path.exists() else meta_path
    if not source_path.exists():
        raise RuntimeError(f"Target conditioning requires pick_place_report.json or meta.json: {episode_dir}")
    meta = json.loads(source_path.read_text(encoding="utf-8"))
    camera_config = meta.get("camera_config", "")
    if not camera_config:
        raise RuntimeError(f"Target conditioning requires camera_config in {source_path}")
    object_pos = np.asarray(meta.get("object_start_pos_m", meta.get("tape_start_positions_m", {}).get(meta.get("object_name", ""), [])), dtype=np.float32)
    goal_pos = np.asarray(meta.get("goal_pos_m", meta.get("object_drop_target_pos_m", [])), dtype=np.float32)
    if object_pos.shape != (3,) or goal_pos.shape != (3,):
        raise RuntimeError(f"Could not read object/goal position for target conditioning from {source_path}")
    return make_target_feature(object_pos, goal_pos, camera_config=camera_config)


def build_samples(
    episode_dirs: Iterable[str | Path],
    *,
    include_last: bool = False,
    target_conditioning: bool = False,
    lookahead_frames: int = 1,
) -> list[BcSample]:
    samples: list[BcSample] = []
    horizon = max(1, int(lookahead_frames))
    for episode_dir_raw in episode_dirs:
        episode_dir = resolve_demo_path(episode_dir_raw)
        pack = load_episode_npz(episode_dir)
        q = np.asarray(pack["joint_rad"], dtype=np.float32)
        recorded_action = np.asarray(pack["action_joint_delta_rad"], dtype=np.float32)
        gripper = (
            np.asarray(pack["gripper_closure"], dtype=np.float32)
            if "gripper_closure" in pack
            else np.zeros(q.shape[0], dtype=np.float32)
        )
        gripper_action = (
            np.asarray(pack["action_gripper_delta"], dtype=np.float32)
            if "action_gripper_delta" in pack
            else np.zeros(q.shape[0], dtype=np.float32)
        )
        reward = (
            np.asarray(pack["reward"], dtype=np.float32)
            if "reward" in pack
            else np.zeros(q.shape[0], dtype=np.float32)
        )
        target_features = (
            np.asarray(pack["target_feature"], dtype=np.float32)
            if bool(target_conditioning) and "target_feature" in pack
            else None
        )
        image_files = [str(item) for item in np.asarray(pack["image_files"]).tolist()]
        target_feature = (
            None
            if target_features is not None
            else target_feature_from_episode(episode_dir, target_conditioning=bool(target_conditioning))
        )
        if q.ndim != 2 or q.shape[1] != ROBOT_DOF:
            raise RuntimeError(f"{episode_dir}/states.npz joint_rad must be Nx6, got {q.shape}")
        if recorded_action.shape != q.shape:
            raise RuntimeError(f"{episode_dir}/states.npz action_joint_delta_rad must be {q.shape}, got {recorded_action.shape}")
        if len(image_files) != q.shape[0]:
            raise RuntimeError(f"{episode_dir}/states.npz image_files length mismatch: {len(image_files)} vs {q.shape[0]}")
        if gripper.shape[0] != q.shape[0] or gripper_action.shape[0] != q.shape[0]:
            raise RuntimeError(f"{episode_dir}/states.npz gripper fields must have {q.shape[0]} frames")
        if reward.shape[0] != q.shape[0]:
            raise RuntimeError(f"{episode_dir}/states.npz reward must have {q.shape[0]} frames")
        if q.shape[0] <= horizon and not include_last:
            continue
        count = q.shape[0] if include_last else max(0, q.shape[0] - horizon)
        for idx in range(count):
            next_idx = min(idx + horizon, q.shape[0] - 1)
            action = (q[next_idx] - q[idx]).astype(np.float32)
            gripper_action_i = float(gripper[next_idx] - gripper[idx])
            samples.append(
                BcSample(
                    episode_dir,
                    image_files[idx],
                    q[idx].copy(),
                    float(gripper[idx]),
                    (
                        target_features[idx].copy()
                        if target_features is not None
                        else np.asarray(target_feature, dtype=np.float32).copy()
                    ),
                    action.copy(),
                    gripper_action_i,
                    float(reward[idx]),
                )
            )
    if not samples:
        raise RuntimeError(f"No usable BC samples. Record more than {horizon} frames per episode.")
    return samples


def compute_bc_stats(
    samples: list[BcSample],
    *,
    action_normalization: str = "standard",
    action_scale_deg: float = DEFAULT_ACTION_SCALE_DEG,
    gripper_action_scale: float = DEFAULT_GRIPPER_ACTION_SCALE,
    target_feature_std_floor: float = DEFAULT_TARGET_FEATURE_STD_FLOOR,
    state_clip: float = DEFAULT_STATE_CLIP,
) -> dict[str, np.ndarray]:
    states = np.stack(
        [
            np.concatenate(
                [
                    sample.joint_rad,
                    np.asarray([sample.gripper_closure], dtype=np.float32),
                    np.asarray(sample.target_feature, dtype=np.float32),
                ]
            )
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)
    actions = np.stack(
        [
            np.concatenate([sample.action_delta_rad, np.asarray([sample.action_gripper_delta], dtype=np.float32)])
            for sample in samples
        ],
        axis=0,
    ).astype(np.float32)
    q_std = states.std(axis=0)
    if states.shape[1] > STATE_DIM:
        q_std[STATE_DIM:] = np.maximum(q_std[STATE_DIM:], float(target_feature_std_floor))
    if action_normalization == "fixed":
        action_mean = np.zeros(actions.shape[1], dtype=np.float32)
        a_std = np.asarray(
            [np.deg2rad(float(action_scale_deg))] * ROBOT_DOF + [float(gripper_action_scale)],
            dtype=np.float32,
        )
    elif action_normalization == "standard":
        action_mean = actions.mean(axis=0)
        a_std = actions.std(axis=0)
    else:
        raise ValueError(f"Unknown action_normalization {action_normalization!r}; expected 'standard' or 'fixed'")
    return {
        "state_mean": states.mean(axis=0),
        "state_std": np.maximum(q_std, 1e-6),
        "action_mean": action_mean.astype(np.float32),
        "action_std": np.maximum(a_std, 1e-6),
        "state_clip": np.asarray([float(state_clip)], dtype=np.float32),
        "action_normalization": np.asarray([0 if action_normalization == "standard" else 1], dtype=np.int64),
    }


def split_samples(samples: list[BcSample], val_ratio: float, seed: int) -> tuple[list[BcSample], list[BcSample]]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(samples))
    val_count = int(round(len(samples) * float(val_ratio)))
    val_count = min(max(val_count, 0), max(0, len(samples) - 1))
    val_idx = set(int(i) for i in order[:val_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in val_idx]
    val = [sample for idx, sample in enumerate(samples) if idx in val_idx]
    return train, val


def split_samples_train_val_test(
    samples: list[BcSample], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[BcSample], list[BcSample], list[BcSample]]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(samples))
    test_count = int(round(len(samples) * float(test_ratio)))
    val_count = int(round(len(samples) * float(val_ratio)))
    test_count = min(max(test_count, 0), max(0, len(samples) - 1))
    val_count = min(max(val_count, 0), max(0, len(samples) - test_count - 1))
    test_idx = set(int(i) for i in order[:test_count])
    val_idx = set(int(i) for i in order[test_count : test_count + val_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in test_idx and idx not in val_idx]
    val = [sample for idx, sample in enumerate(samples) if idx in val_idx]
    test = [sample for idx, sample in enumerate(samples) if idx in test_idx]
    return train, val, test


class Fr5BcDataset(Dataset):
    def __init__(
        self,
        samples: list[BcSample],
        *,
        image_size: int,
        stats: dict[str, np.ndarray],
        cache_images: bool = False,
        preload_images: bool = False,
        load_images: bool = True,
    ):
        self.samples = list(samples)
        self.image_size = int(image_size)
        self.stats = {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()}
        self.load_images = bool(load_images)
        self._cv2 = None
        self._image_cache: dict[tuple[str, str], np.ndarray] | None = {} if bool(cache_images and load_images) else None
        if self._image_cache is not None and bool(preload_images):
            for sample in self.samples:
                key = (sample.episode_dir.as_posix(), sample.image_file)
                if key not in self._image_cache:
                    self._image_cache[key] = self._read_rgb_uint8(sample.episode_dir / sample.image_file)

    def __len__(self) -> int:
        return len(self.samples)

    def _cv2_module(self):
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def _read_rgb_uint8(self, path: Path) -> np.ndarray:
        cv2 = self._cv2_module()
        bgr = cv2.imread(path.as_posix(), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read BC image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return np.transpose(np.ascontiguousarray(rgb, dtype=np.uint8), (2, 0, 1))

    def _read_rgb(self, sample: BcSample) -> np.ndarray:
        image_path = sample.episode_dir / sample.image_file
        key = (sample.episode_dir.as_posix(), sample.image_file)
        if self._image_cache is not None:
            image_u8 = self._image_cache.get(key)
            if image_u8 is None:
                image_u8 = self._read_rgb_uint8(image_path)
                self._image_cache[key] = image_u8
        else:
            image_u8 = self._read_rgb_uint8(image_path)
        return image_u8.astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[int(idx)]
        image = self._read_rgb(sample) if self.load_images else np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        state_raw = np.concatenate([sample.joint_rad, np.asarray([sample.gripper_closure], dtype=np.float32)]).astype(
            np.float32
        )
        if self.stats["state_mean"].shape[0] > STATE_DIM:
            state_raw = np.concatenate([state_raw, np.asarray(sample.target_feature, dtype=np.float32)])
        action_raw = np.concatenate(
            [sample.action_delta_rad, np.asarray([sample.action_gripper_delta], dtype=np.float32)]
        ).astype(np.float32)
        state = (state_raw - self.stats["state_mean"]) / self.stats["state_std"]
        state_clip = float(np.asarray(self.stats.get("state_clip", [DEFAULT_STATE_CLIP]), dtype=np.float32)[0])
        if state_clip > 0.0:
            state = np.clip(state, -state_clip, state_clip)
        action = (action_raw - self.stats["action_mean"]) / self.stats["action_std"]
        action = np.clip(action, -1.0, 1.0)
        return {
            "image": torch.from_numpy(image),
            "state": torch.from_numpy(state.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
            "reward": torch.tensor([float(sample.reward)], dtype=torch.float32),
        }


class SmallImageJointPolicy(nn.Module):
    def __init__(self, *, action_dim: int = STATE_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([img_feat, state_feat], dim=-1))


class MediumImageJointPolicy(nn.Module):
    def __init__(self, *, action_dim: int = STATE_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([img_feat, state_feat], dim=-1))


class StateOnlyMlpPolicy(nn.Module):
    def __init__(self, *, action_dim: int = STATE_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class LateFusionMlpPolicy(nn.Module):
    def __init__(self, *, action_dim: int = STATE_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        return self.head(torch.cat([img_feat, state], dim=-1))


class SpatialSoftmax(nn.Module):
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature.shape
        flat = feature.reshape(b, c, h * w)
        prob = torch.softmax(flat, dim=-1)
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feature.device, dtype=feature.dtype),
            torch.linspace(-1.0, 1.0, w, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        exp_x = torch.sum(prob * pos_x.reshape(1, 1, h * w), dim=-1)
        exp_y = torch.sum(prob * pos_y.reshape(1, 1, h * w), dim=-1)
        return torch.cat([exp_x, exp_y], dim=-1)


class SpatialSoftmaxPolicy(nn.Module):
    def __init__(self, *, action_dim: int = STATE_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial = SpatialSoftmax()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 2 + 96, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        keypoints = self.spatial(self.conv(image))
        state_feat = self.state_encoder(state)
        return self.head(torch.cat([keypoints, state_feat], dim=-1))


class RewardAugmentedBcPolicy(nn.Module):
    def __init__(self, *, action_dim: int = STATE_DIM, state_dim: int = STATE_DIM):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 + 96, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(128, action_dim)
        self.reward_head = nn.Linear(128, 1)

    def forward_features(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        state_feat = self.state_encoder(state)
        return self.trunk(torch.cat([img_feat, state_feat], dim=-1))

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.action_head(self.forward_features(image, state)))

    def forward_with_aux(self, image: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.forward_features(image, state)
        return torch.tanh(self.action_head(feat)), self.reward_head(feat)


POLICY_REGISTRY = {
    "cnn_small": SmallImageJointPolicy,
    "cnn_medium": MediumImageJointPolicy,
    "late_fusion": LateFusionMlpPolicy,
    "state_mlp": StateOnlyMlpPolicy,
    "spatial_softmax": SpatialSoftmaxPolicy,
    "reward_bc": RewardAugmentedBcPolicy,
    "SmallImageJointPolicy": SmallImageJointPolicy,
}


def policy_model_types() -> list[str]:
    return ["cnn_small", "cnn_medium", "late_fusion", "state_mlp", "spatial_softmax", "reward_bc"]


def recommended_policy_model_types() -> list[str]:
    return ["cnn_small", "state_mlp", "spatial_softmax"]


def create_policy_model(
    model_type: str = DEFAULT_MODEL_TYPE,
    *,
    state_dim: int = STATE_DIM,
    action_dim: int = STATE_DIM,
) -> nn.Module:
    if model_type not in POLICY_REGISTRY:
        raise ValueError(f"Unknown policy model_type {model_type!r}. Available: {policy_model_types()}")
    return POLICY_REGISTRY[model_type](state_dim=int(state_dim), action_dim=int(action_dim))


def save_policy_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    stats: dict[str, np.ndarray],
    image_size: int,
    model_type: str = DEFAULT_MODEL_TYPE,
    meta: dict,
) -> Path:
    path = resolve_demo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "stats": {key: np.asarray(value, dtype=np.float32) for key, value in stats.items()},
        "image_size": int(image_size),
        "action_mode": "joint_delta_rad_plus_gripper_delta",
        "model_class": model.__class__.__name__,
        "model_type": str(model_type),
        "state_dim": int(np.asarray(stats["state_mean"]).shape[0]),
        "action_dim": STATE_DIM,
        "meta": dict(meta),
    }
    torch.save(payload, path.as_posix())
    return path


def load_policy_checkpoint(path: str | Path, *, device: str | torch.device):
    path = resolve_demo_path(path)
    payload = torch.load(path.as_posix(), map_location=device, weights_only=False)
    model_type = str(payload.get("model_type", payload.get("model_class", DEFAULT_MODEL_TYPE)))
    model = create_policy_model(
        model_type,
        state_dim=int(payload.get("state_dim", STATE_DIM)),
        action_dim=int(payload.get("action_dim", STATE_DIM)),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    stats = {key: np.asarray(value, dtype=np.float32) for key, value in payload["stats"].items()}
    image_size = int(payload.get("image_size", 128))
    return model, stats, image_size, payload


def preprocess_rgb_for_policy(rgb: np.ndarray, *, image_size: int) -> torch.Tensor:
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError(f"Policy RGB image must be HxWx3, got {rgb.shape}")
    resized = cv2.resize(rgb[..., :3], (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    image = resized.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def normalize_joint(
    joint_rad: np.ndarray,
    stats: dict[str, np.ndarray],
    gripper_closure: float = 0.0,
    target_feature: np.ndarray | None = None,
) -> torch.Tensor:
    q = np.asarray(joint_rad, dtype=np.float32)
    state = np.concatenate([q, np.asarray([gripper_closure], dtype=np.float32)]).astype(np.float32)
    if "state_mean" in stats:
        expected = int(np.asarray(stats["state_mean"]).shape[0])
        if expected > state.shape[0]:
            feature = (
                np.zeros(expected - state.shape[0], dtype=np.float32)
                if target_feature is None
                else np.asarray(target_feature, dtype=np.float32)
            )
            if feature.shape[0] != expected - state.shape[0]:
                raise RuntimeError(f"target_feature must have {expected - state.shape[0]} values, got {feature.shape}")
            state = np.concatenate([state, feature.astype(np.float32)])
        state = (state - stats["state_mean"]) / stats["state_std"]
        state_clip = float(np.asarray(stats.get("state_clip", [DEFAULT_STATE_CLIP]), dtype=np.float32)[0])
        if state_clip > 0.0:
            state = np.clip(state, -state_clip, state_clip)
    else:
        legacy = (q - stats["joint_mean"]) / stats["joint_std"]
        return torch.from_numpy(legacy.astype(np.float32)).unsqueeze(0)
    return torch.from_numpy(state.astype(np.float32)).unsqueeze(0)


def denormalize_action(action_norm: torch.Tensor, stats: dict[str, np.ndarray]) -> np.ndarray:
    action = action_norm.detach().cpu().numpy()[0].astype(np.float32)
    action = np.clip(action, -1.0, 1.0)
    return action * stats["action_std"] + stats["action_mean"]
