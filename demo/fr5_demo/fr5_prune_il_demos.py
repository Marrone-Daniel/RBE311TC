from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from fr5_il_dataset import DEFAULT_IL_DEMO_DIR, TARGET_FEATURE_DIM, resolve_demo_path


REQUIRED_ARRAYS = {
    "joint_rad": (None, 6),
    "action_joint_delta_rad": (None, 6),
    "gripper_closure": (None,),
    "action_gripper_delta": (None,),
    "image_files": (None,),
}


def _shape_ok(shape: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    if len(shape) != len(expected):
        return False
    return all(exp is None or got == exp for got, exp in zip(shape, expected))


def inspect_episode(path: Path, *, require_target: bool, require_reward: bool, require_success: bool) -> list[str]:
    reasons: list[str] = []
    states_path = path / "states.npz"
    if not states_path.exists():
        return ["missing states.npz"]
    try:
        pack = np.load(states_path, allow_pickle=True)
    except Exception as exc:
        return [f"cannot load states.npz: {exc}"]

    frame_count = None
    for key, expected_shape in REQUIRED_ARRAYS.items():
        if key not in pack:
            reasons.append(f"missing array {key}")
            continue
        arr = pack[key]
        if not _shape_ok(tuple(arr.shape), expected_shape):
            reasons.append(f"bad shape {key}: {arr.shape}")
            continue
        if frame_count is None:
            frame_count = int(arr.shape[0])
        elif int(arr.shape[0]) != frame_count:
            reasons.append(f"frame count mismatch {key}: {arr.shape[0]} vs {frame_count}")

    if frame_count is not None and frame_count < 2:
        reasons.append(f"too few frames: {frame_count}")

    if require_target:
        if "target_feature" not in pack:
            reasons.append("missing target_feature")
        elif tuple(pack["target_feature"].shape) != (frame_count, TARGET_FEATURE_DIM):
            reasons.append(f"bad shape target_feature: {pack['target_feature'].shape}")

    if require_reward:
        if "reward" not in pack:
            reasons.append("missing reward")
        elif tuple(pack["reward"].shape) != (frame_count,):
            reasons.append(f"bad shape reward: {pack['reward'].shape}")

    if "image_files" in pack:
        missing_images = []
        for item in np.asarray(pack["image_files"]).tolist()[:5]:
            if not (path / str(item)).exists():
                missing_images.append(str(item))
        if missing_images:
            reasons.append(f"missing referenced images, examples={missing_images}")

    meta_path = path / "pick_place_report.json"
    if not meta_path.exists():
        meta_path = path / "meta.json"
    if require_target or require_success:
        if not meta_path.exists():
            reasons.append("missing pick_place_report.json/meta.json")
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                reasons.append(f"cannot parse metadata: {exc}")
                meta = {}
            if require_target:
                for key in ("camera_config", "object_start_pos_m", "goal_pos_m"):
                    if key not in meta:
                        reasons.append(f"missing metadata {key}")
            if require_success and meta.get("task_success") is not True:
                reasons.append(f"task_success is not true: {meta.get('task_success')}")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Move obsolete or incompatible FR5 IL demo episodes out of the training directory.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--quarantine-dir", type=str, default="")
    parser.add_argument("--require-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_dir = resolve_demo_path(args.data_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = (
        resolve_demo_path(args.quarantine_dir)
        if args.quarantine_dir
        else data_dir.parent / f"{data_dir.name}_disabled_{stamp}"
    )

    rows: list[dict] = []
    for episode_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if episode_dir.name.startswith("_"):
            rows.append({"episode": episode_dir.name, "status": "skip_internal_dir", "reasons": []})
            continue
        reasons = inspect_episode(
            episode_dir,
            require_target=bool(args.require_target),
            require_reward=bool(args.require_reward),
            require_success=bool(args.require_success),
        )
        status = "move" if reasons else "keep"
        rows.append({"episode": episode_dir.name, "status": status, "reasons": reasons})

    to_move = [row for row in rows if row["status"] == "move"]
    if to_move and not args.dry_run:
        quarantine.mkdir(parents=True, exist_ok=True)
        for row in to_move:
            src = data_dir / row["episode"]
            dst = quarantine / row["episode"]
            if dst.exists():
                raise RuntimeError(f"Refusing to overwrite existing quarantine episode: {dst}")
            shutil.move(src.as_posix(), dst.as_posix())

    report_path = data_dir.parent / f"il_demo_prune_report_{stamp}.json"
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": data_dir.as_posix(),
        "quarantine_dir": quarantine.as_posix(),
        "dry_run": bool(args.dry_run),
        "require_target": bool(args.require_target),
        "require_reward": bool(args.require_reward),
        "require_success": bool(args.require_success),
        "kept": sum(1 for row in rows if row["status"] == "keep"),
        "moved": len(to_move) if not args.dry_run else 0,
        "would_move": len(to_move),
        "episodes": rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Checked {len(rows)} episode dirs.")
    print(f"keep={report['kept']} would_move={report['would_move']} moved={report['moved']} dry_run={report['dry_run']}")
    if to_move:
        print(f"quarantine={quarantine}")
        for row in to_move[:20]:
            print(f"  {row['episode']}: {', '.join(row['reasons'])}")
        if len(to_move) > 20:
            print(f"  ... {len(to_move) - 20} more")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
