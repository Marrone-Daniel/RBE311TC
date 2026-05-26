from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fr5_il_dataset import (
    DEFAULT_IL_DEMO_DIR,
    DEFAULT_POLICY_DIR,
    policy_model_types,
    recommended_policy_model_types,
    resolve_demo_path,
)


def load_best_row(history_path: Path) -> dict:
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return {}
    return min(history, key=lambda row: float(row.get("val_loss", float("inf"))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all selected FR5 BC policy architectures with one command.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--models", nargs="+", choices=policy_model_types(), default=recommended_policy_model_types())
    parser.add_argument("--out-dir", type=str, default=DEFAULT_POLICY_DIR.as_posix())
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lookahead-frames", type=int, default=1)
    parser.add_argument("--action-normalization", choices=["fixed", "standard"], default="fixed")
    parser.add_argument("--action-scale-deg", type=float, default=18.0)
    parser.add_argument("--gripper-action-scale", type=float, default=0.1)
    parser.add_argument("--target-feature-std-floor", type=float, default=0.05)
    parser.add_argument("--state-clip", type=float, default=5.0)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preload-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--test-every", type=int, default=0)
    parser.add_argument("--reward-loss-weight", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--target-conditioning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if int(args.lookahead_frames) < 1:
        raise RuntimeError("--lookahead-frames must be >= 1")

    out_dir = resolve_demo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    train_script = Path(__file__).resolve().parent / "fr5_train_bc.py"

    summary: list[dict] = []
    print("Training model batch:")
    print(f"  models={args.models}")
    print(f"  out_dir={out_dir}")
    print(f"  run_name={stamp}")
    print(f"  optimizer={args.optimizer} lr={float(args.lr):.3g} weight_decay={float(args.weight_decay):.3g}")
    print(f"  warmup_epochs={float(args.warmup_epochs):.3g} warmup_steps={int(args.warmup_steps)}")
    print(f"  lookahead_frames={int(args.lookahead_frames)}")
    print(f"  action_normalization={args.action_normalization} action_scale_deg={float(args.action_scale_deg):.3g}")
    print(
        f"  cache_images={bool(args.cache_images)} preload_images={bool(args.preload_images)} "
        f"amp={bool(args.amp)} early_stopping_patience={int(args.early_stopping_patience)}"
    )
    print(f"  split train/val/test = remaining/{args.val_ratio}/{args.test_ratio}")

    for model_type in args.models:
        policy_path = out_dir / f"fr5_bc_{model_type}_{stamp}.pt"
        cmd = [
            sys.executable,
            train_script.as_posix(),
            "--data-dir",
            args.data_dir,
            "--model-type",
            model_type,
            "--out",
            policy_path.as_posix(),
            "--image-size",
            str(int(args.image_size)),
            "--batch-size",
            str(int(args.batch_size)),
            "--epochs",
            str(int(args.epochs)),
            "--optimizer",
            args.optimizer,
            "--lr",
            str(float(args.lr)),
            "--weight-decay",
            str(float(args.weight_decay)),
            "--warmup-epochs",
            str(float(args.warmup_epochs)),
            "--warmup-steps",
            str(int(args.warmup_steps)),
            "--lookahead-frames",
            str(int(args.lookahead_frames)),
            "--action-normalization",
            args.action_normalization,
            "--action-scale-deg",
            str(float(args.action_scale_deg)),
            "--gripper-action-scale",
            str(float(args.gripper_action_scale)),
            "--target-feature-std-floor",
            str(float(args.target_feature_std_floor)),
            "--state-clip",
            str(float(args.state_clip)),
            "--grad-clip-norm",
            str(float(args.grad_clip_norm)),
            "--early-stopping-patience",
            str(int(args.early_stopping_patience)),
            "--min-delta",
            str(float(args.min_delta)),
            "--test-every",
            str(int(args.test_every)),
            "--reward-loss-weight",
            str(float(args.reward_loss_weight)),
            "--val-ratio",
            str(float(args.val_ratio)),
            "--test-ratio",
            str(float(args.test_ratio)),
            "--seed",
            str(int(args.seed)),
            "--device",
            args.device,
            "--num-workers",
            str(int(args.num_workers)),
            "--log-interval",
            str(int(args.log_interval)),
        ]
        cmd.append("--cache-images" if args.cache_images else "--no-cache-images")
        cmd.append("--preload-images" if args.preload_images else "--no-preload-images")
        cmd.append("--amp" if args.amp else "--no-amp")
        if args.target_conditioning:
            cmd.append("--target-conditioning")
        if args.episodes:
            cmd.append("--episodes")
            cmd.extend(args.episodes)

        print(f"\n=== Training {model_type} ===", flush=True)
        print(" ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2].as_posix(), check=False)
        status = {
            "model_type": model_type,
            "policy": policy_path.as_posix(),
            "history": policy_path.with_suffix(".history.json").as_posix(),
            "returncode": int(proc.returncode),
        }
        if proc.returncode == 0 and policy_path.with_suffix(".history.json").exists():
            best = load_best_row(policy_path.with_suffix(".history.json"))
            status.update(
                {
                    "best_epoch": best.get("epoch"),
                    "best_val_loss": best.get("val_loss"),
                    "best_val_arm_mae_deg": best.get("val_arm_mae_deg"),
                    "test_loss_at_best_epoch": best.get("test_loss"),
                    "test_arm_mae_deg_at_best_epoch": best.get("test_arm_mae_deg"),
                    "test_gripper_mae_at_best_epoch": best.get("test_gripper_mae"),
                }
            )
        summary.append(status)
        if proc.returncode != 0 and not args.keep_going:
            break

    summary_path = out_dir / f"fr5_bc_model_comparison_{stamp}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "run_name": stamp,
                "models": args.models,
                "data_dir": args.data_dir,
                "episodes": args.episodes,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "image_size": int(args.image_size),
                "optimizer": args.optimizer,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "warmup_epochs": float(args.warmup_epochs),
                "warmup_steps": int(args.warmup_steps),
                "lookahead_frames": int(args.lookahead_frames),
                "action_normalization": str(args.action_normalization),
                "action_scale_deg": float(args.action_scale_deg),
                "gripper_action_scale": float(args.gripper_action_scale),
                "target_feature_std_floor": float(args.target_feature_std_floor),
                "state_clip": float(args.state_clip),
                "cache_images": bool(args.cache_images),
                "preload_images": bool(args.preload_images),
                "amp": bool(args.amp),
                "grad_clip_norm": float(args.grad_clip_norm),
                "early_stopping_patience": int(args.early_stopping_patience),
                "min_delta": float(args.min_delta),
                "test_every": int(args.test_every),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(args.test_ratio),
                "target_conditioning": bool(args.target_conditioning),
                "reward_loss_weight": float(args.reward_loss_weight),
                "seed": int(args.seed),
                "results": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nModel comparison summary:")
    for row in summary:
        print(
            f"  {row['model_type']}: rc={row['returncode']} best_epoch={row.get('best_epoch')} "
            f"val_loss={row.get('best_val_loss')} test_loss={row.get('test_loss_at_best_epoch')} "
            f"test_arm_mae_deg={row.get('test_arm_mae_deg_at_best_epoch')}",
            flush=True,
        )
    print(f"Saved comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
