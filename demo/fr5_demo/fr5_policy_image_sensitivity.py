from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from arm_control import DEFAULT_CONFIG, load_config, resolve_demo_path
from fr5_il_dataset import DEFAULT_IL_DEMO_DIR, DEFAULT_POLICY_DIR, list_episode_dirs, load_policy_checkpoint
from fr5_policy_rollout import choose_device, policy_requires_target_feature, predict_delta_rad, read_episode_rgb, target_feature_from_episode_dir


DEFAULT_POLICY = DEFAULT_POLICY_DIR / "fr5_bc_last.pt"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "policy_diagnostics"


def load_episode_first_frame(episode_dir: Path) -> tuple[np.ndarray, float, str, np.ndarray]:
    pack = np.load((episode_dir / "states.npz").as_posix(), allow_pickle=True)
    q0 = np.asarray(pack["joint_rad"][0], dtype=np.float32)
    gripper = float(np.asarray(pack["gripper_closure"], dtype=np.float32)[0]) if "gripper_closure" in pack else 0.0
    image_file = str(np.asarray(pack["image_files"]).tolist()[0])
    expert = np.asarray(pack["action_joint_delta_rad"][0], dtype=np.float32)
    return q0, gripper, image_file, expert


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether a policy changes its first action when the RGB image changes. "
            "Low predicted variance with high expert variance means the policy is behaving like an open-loop controller."
        )
    )
    parser.add_argument("--policy", type=str, default=DEFAULT_POLICY.as_posix())
    parser.add_argument("--data-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG.as_posix())
    parser.add_argument("--same-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT.as_posix())
    args = parser.parse_args()

    device = choose_device(args.device)
    model, stats, image_size, payload = load_policy_checkpoint(args.policy, device=device)
    episode_dirs = list_episode_dirs(args.data_dir, args.episodes)[: max(1, int(args.max_episodes))]
    config = load_config(args.config)
    shared_q = np.asarray(config["initial_qpos"], dtype=np.float32)

    rows: list[dict] = []
    pred_actions: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    for episode_dir in episode_dirs:
        q0, gripper, image_file, expert = load_episode_first_frame(episode_dir)
        q_in = shared_q if bool(args.same_state) else q0
        rgb = read_episode_rgb(episode_dir, image_file)
        target_feature = target_feature_from_episode_dir(episode_dir, args.config) if policy_requires_target_feature(stats) else None
        pred_full = predict_delta_rad(model, stats, image_size, device, rgb, q_in, gripper, target_feature=target_feature)
        pred = np.asarray(pred_full[:6], dtype=np.float32)
        pred_actions.append(pred)
        expert_actions.append(expert)
        rows.append(
            {
                "episode": episode_dir.as_posix(),
                "image_file": image_file,
                "same_state": bool(args.same_state),
                **{f"pred_j{i+1}_deg": float(np.rad2deg(pred[i])) for i in range(6)},
                **{f"expert_j{i+1}_deg": float(np.rad2deg(expert[i])) for i in range(6)},
            }
        )

    pred_arr = np.stack(pred_actions, axis=0) if pred_actions else np.zeros((0, 6), dtype=np.float32)
    expert_arr = np.stack(expert_actions, axis=0) if expert_actions else np.zeros((0, 6), dtype=np.float32)
    pred_std = np.rad2deg(np.std(pred_arr, axis=0)) if pred_arr.size else np.zeros(6)
    expert_std = np.rad2deg(np.std(expert_arr, axis=0)) if expert_arr.size else np.zeros(6)
    ratio = np.divide(pred_std, expert_std, out=np.full_like(pred_std, np.nan), where=expert_std > 1e-6)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": resolve_demo_path(args.policy).as_posix(),
        "episodes": len(rows),
        "same_state": bool(args.same_state),
        "pred_action_std_deg": pred_std.round(6).tolist(),
        "expert_action_std_deg": expert_std.round(6).tolist(),
        "pred_to_expert_std_ratio": ratio.round(6).tolist(),
        "mean_ratio": float(np.nanmean(ratio)) if ratio.size else 0.0,
        "interpretation": (
            "If pred_to_expert_std_ratio is near 0, the policy barely reacts to changed RGB and is effectively open-loop. "
            "Ratios closer to 1 mean the policy action changes with the same scale as the demonstrations."
        ),
    }

    out_dir = resolve_demo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"image_sensitivity_{Path(args.policy).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved JSON: {json_path}", flush=True)
    print(f"Saved CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
