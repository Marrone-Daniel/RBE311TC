from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from fr5_il_dataset import (
    DEFAULT_IL_DEMO_DIR,
    DEFAULT_POLICY_DIR,
    Fr5BcDataset,
    build_samples,
    compute_bc_stats,
    create_policy_model,
    list_episode_dirs,
    policy_model_types,
    save_policy_checkpoint,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def run_epoch(
    model,
    loader,
    *,
    device,
    stats: dict[str, np.ndarray],
    optimizer=None,
    scheduler=None,
    scaler=None,
    amp: bool = False,
    grad_clip_norm: float = 0.0,
    phase: str = "train",
    epoch: int | None = None,
    log_interval: int = 0,
    reward_loss_weight: float = 0.1,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.SmoothL1Loss()
    total = 0.0
    arm_abs = 0.0
    grip_abs = 0.0
    reward_abs = 0.0
    reward_total = 0.0
    count = 0
    total_batches = len(loader)
    for batch_idx, batch in enumerate(loader, start=1):
        image = batch["image"].to(device=device, dtype=torch.float32)
        state = batch["state"].to(device=device, dtype=torch.float32)
        target = batch["action"].to(device=device, dtype=torch.float32)
        reward_target = batch["reward"].to(device=device, dtype=torch.float32)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
            if hasattr(model, "forward_with_aux"):
                pred, reward_pred = model.forward_with_aux(image, state)
                action_loss = loss_fn(pred, target)
                reward_loss = loss_fn(reward_pred, reward_target)
                loss = action_loss + float(reward_loss_weight) * reward_loss
            else:
                pred = model(image, state)
                reward_pred = None
                reward_loss = None
                loss = loss_fn(pred, target)
        pred = pred.float()
        if train and scaler is not None and bool(amp and device.type == "cuda"):
            scaler.scale(loss).backward()
            if float(grad_clip_norm) > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
        elif train:
            loss.backward()
            if float(grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        total += float(loss.detach().cpu()) * image.shape[0]
        if reward_loss is not None:
            reward_total += float(reward_loss.detach().cpu()) * image.shape[0]
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        action_std = np.asarray(stats["action_std"], dtype=np.float32)
        pred_raw = pred_np * action_std + np.asarray(stats["action_mean"], dtype=np.float32)
        target_raw = target_np * action_std + np.asarray(stats["action_mean"], dtype=np.float32)
        arm_abs += float(np.sum(np.abs(pred_raw[:, :6] - target_raw[:, :6])))
        grip_abs += float(np.sum(np.abs(pred_raw[:, 6] - target_raw[:, 6])))
        if reward_pred is not None:
            reward_abs += float(torch.sum(torch.abs(reward_pred.detach() - reward_target)).cpu())
        count += int(image.shape[0])
        if log_interval > 0 and (batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches):
            prefix = f"epoch={epoch:03d} " if epoch is not None else ""
            print(f"{prefix}{phase} batch={batch_idx}/{total_batches} samples={count}", flush=True)
    return {
        "loss": total / max(1, count),
        "arm_mae_deg": float(np.rad2deg(arm_abs / max(1, count * 6))),
        "gripper_mae": grip_abs / max(1, count),
        "reward_loss": reward_total / max(1, count),
        "reward_mae": reward_abs / max(1, count),
    }


def split_episode_dirs_train_val_test(episode_dirs, val_ratio: float, test_ratio: float, seed: int):
    episode_dirs = list(episode_dirs)
    if len(episode_dirs) < 2:
        return episode_dirs, [], []
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(episode_dirs))
    test_count = int(round(len(episode_dirs) * float(test_ratio)))
    val_count = int(round(len(episode_dirs) * float(val_ratio)))
    test_count = min(max(test_count, 0), max(0, len(episode_dirs) - 1))
    val_count = min(max(val_count, 0), max(0, len(episode_dirs) - test_count - 1))
    test_idx = set(int(i) for i in order[:test_count])
    val_idx = set(int(i) for i in order[test_count : test_count + val_count])
    train = [path for idx, path in enumerate(episode_dirs) if idx not in test_idx and idx not in val_idx]
    val = [path for idx, path in enumerate(episode_dirs) if idx in val_idx]
    test = [path for idx, path in enumerate(episode_dirs) if idx in test_idx]
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small behavior-cloning policy from FR5 IL episodes.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_IL_DEMO_DIR.as_posix())
    parser.add_argument("--episodes", nargs="*", default=None, help="Optional episode dirs/names. Defaults to all under --data-dir.")
    parser.add_argument("--out", type=str, default=(DEFAULT_POLICY_DIR / "fr5_bc_last.pt").as_posix())
    parser.add_argument("--model-type", choices=policy_model_types(), default="cnn_small")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Linearly warm up LR for this many epochs.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Override warmup length in optimizer steps. 0 uses --warmup-epochs.")
    parser.add_argument(
        "--lookahead-frames",
        type=int,
        default=1,
        help="Train action labels as q[t+N]-q[t]. Rollout executes this as 1/N per frame and replans every frame.",
    )
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
    parser.add_argument("--test-every", type=int, default=0, help="Run test split every N epochs. 0 only runs final test.")
    parser.add_argument("--reward-loss-weight", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument(
        "--target-conditioning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append projected object/goal camera features to the policy state so actions can adapt to tape image position.",
    )
    parser.add_argument(
        "--split-by-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split train/val/test by whole episodes. This avoids adjacent frames from one trajectory leaking across splits.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20, help="Print progress every N batches. Use 0 to disable.")
    args = parser.parse_args()
    if int(args.lookahead_frames) < 1:
        raise RuntimeError("--lookahead-frames must be >= 1")

    torch.manual_seed(int(args.seed))
    device = choose_device(args.device)
    episode_dirs = list_episode_dirs(args.data_dir, args.episodes)
    if bool(args.split_by_episode):
        train_episode_dirs, val_episode_dirs, test_episode_dirs = split_episode_dirs_train_val_test(
            episode_dirs, float(args.val_ratio), float(args.test_ratio), int(args.seed)
        )
        train_samples = build_samples(
            train_episode_dirs,
            target_conditioning=bool(args.target_conditioning),
            lookahead_frames=int(args.lookahead_frames),
        )
        val_samples = (
            build_samples(val_episode_dirs, target_conditioning=bool(args.target_conditioning), lookahead_frames=int(args.lookahead_frames))
            if val_episode_dirs
            else []
        )
        test_samples = (
            build_samples(test_episode_dirs, target_conditioning=bool(args.target_conditioning), lookahead_frames=int(args.lookahead_frames))
            if test_episode_dirs
            else []
        )
    else:
        from fr5_il_dataset import split_samples_train_val_test

        samples = build_samples(
            episode_dirs,
            target_conditioning=bool(args.target_conditioning),
            lookahead_frames=int(args.lookahead_frames),
        )
        train_samples, val_samples, test_samples = split_samples_train_val_test(
            samples, float(args.val_ratio), float(args.test_ratio), int(args.seed)
        )
        train_episode_dirs = episode_dirs
        val_episode_dirs = []
        test_episode_dirs = []
    stats = compute_bc_stats(
        train_samples,
        action_normalization=str(args.action_normalization),
        action_scale_deg=float(args.action_scale_deg),
        gripper_action_scale=float(args.gripper_action_scale),
        target_feature_std_floor=float(args.target_feature_std_floor),
        state_clip=float(args.state_clip),
    )

    load_images = str(args.model_type) != "state_mlp"
    train_ds = Fr5BcDataset(
        train_samples,
        image_size=int(args.image_size),
        stats=stats,
        cache_images=bool(args.cache_images),
        preload_images=bool(args.preload_images),
        load_images=load_images,
    )
    val_ds = (
        Fr5BcDataset(
            val_samples,
            image_size=int(args.image_size),
            stats=stats,
            cache_images=bool(args.cache_images),
            preload_images=bool(args.preload_images),
            load_images=load_images,
        )
        if val_samples
        else None
    )
    test_ds = (
        Fr5BcDataset(
            test_samples,
            image_size=int(args.image_size),
            stats=stats,
            cache_images=bool(args.cache_images),
            preload_images=bool(args.preload_images),
            load_images=load_images,
        )
        if test_samples
        else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.num_workers) > 0,
        )
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=int(args.num_workers) > 0,
        )
        if test_ds is not None
        else None
    )

    model = create_policy_model(args.model_type, state_dim=int(stats["state_mean"].shape[0])).to(device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0:
        warmup_steps = int(round(float(args.warmup_epochs) * max(1, len(train_loader))))
    scheduler = None
    if warmup_steps > 1:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)),
        )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    print("BC training setup:")
    print(f"  episodes={len(episode_dirs)}")
    print(
        f"  episode split train/val/test={len(train_episode_dirs)}/{len(val_episode_dirs)}/{len(test_episode_dirs)} "
        f"split_by_episode={bool(args.split_by_episode)}"
    )
    print(f"  target_conditioning={bool(args.target_conditioning)} state_dim={int(stats['state_mean'].shape[0])}")
    print(
        f"  lookahead_frames={int(args.lookahead_frames)} "
        f"execution_lead_at_10hz={int(args.lookahead_frames) / 10.0:.3f}s"
    )
    print(f"  samples train/val/test={len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    print(f"  model_type={args.model_type}")
    print(f"  optimizer={args.optimizer} lr={float(args.lr):.3g} weight_decay={float(args.weight_decay):.3g}")
    print(f"  warmup_steps={warmup_steps}")
    print(
        f"  action_normalization={args.action_normalization} action_scale_deg={float(args.action_scale_deg):.3g} "
        f"gripper_action_scale={float(args.gripper_action_scale):.3g}"
    )
    print(f"  target_feature_std_floor={float(args.target_feature_std_floor):.3g} state_clip={float(args.state_clip):.3g}")
    print(
        f"  cache_images={bool(args.cache_images)} preload_images={bool(args.preload_images)} "
        f"load_images={bool(load_images)} amp={bool(args.amp and device.type == 'cuda')} "
        f"grad_clip_norm={float(args.grad_clip_norm):.3g}"
    )
    print(f"  device={device}")
    print(f"  image_size={int(args.image_size)}", flush=True)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            stats=stats,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp=bool(args.amp),
            grad_clip_norm=float(args.grad_clip_norm),
            phase="train",
            epoch=epoch,
            log_interval=int(args.log_interval),
            reward_loss_weight=float(args.reward_loss_weight),
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                device=device,
                stats=stats,
                phase="val",
                epoch=epoch,
                log_interval=int(args.log_interval),
                reward_loss_weight=float(args.reward_loss_weight),
            )
            if val_loader is not None
            else train_metrics
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_arm_mae_deg": train_metrics["arm_mae_deg"],
            "val_arm_mae_deg": val_metrics["arm_mae_deg"],
            "train_gripper_mae": train_metrics["gripper_mae"],
            "val_gripper_mae": val_metrics["gripper_mae"],
            "train_reward_loss": train_metrics["reward_loss"],
            "val_reward_loss": val_metrics["reward_loss"],
            "train_reward_mae": train_metrics["reward_mae"],
            "val_reward_mae": val_metrics["reward_mae"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        run_test_this_epoch = (
            test_loader is not None
            and int(args.test_every) > 0
            and (epoch == 1 or epoch % int(args.test_every) == 0 or epoch == int(args.epochs))
        )
        if run_test_this_epoch:
            test_metrics = run_epoch(
                model,
                test_loader,
                device=device,
                stats=stats,
                phase="test",
                epoch=epoch,
                log_interval=int(args.log_interval),
                reward_loss_weight=float(args.reward_loss_weight),
            )
            row.update(
                {
                    "test_loss": test_metrics["loss"],
                    "test_arm_mae_deg": test_metrics["arm_mae_deg"],
                    "test_gripper_mae": test_metrics["gripper_mae"],
                    "test_reward_loss": test_metrics["reward_loss"],
                    "test_reward_mae": test_metrics["reward_mae"],
                }
            )
        history.append(row)
        if val_metrics["loss"] < best_val - float(args.min_delta):
            best_val = val_metrics["loss"]
            best_epoch = int(epoch)
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        msg = (
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} val_loss={row['val_loss']:.6f} "
            f"train_arm_mae={row['train_arm_mae_deg']:.3f}deg val_arm_mae={row['val_arm_mae_deg']:.3f}deg"
        )
        if "test_loss" in row:
            msg += f" test_loss={row['test_loss']:.6f} test_arm_mae={row['test_arm_mae_deg']:.3f}deg"
        print(msg, flush=True)
        if int(args.early_stopping_patience) > 0 and stale_epochs >= int(args.early_stopping_patience):
            print(
                f"Early stopping at epoch={epoch}; best_epoch={best_epoch} best_val_loss={best_val:.6f}",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test_metrics = (
        run_epoch(
            model,
            test_loader,
            device=device,
            stats=stats,
            phase="final_test",
            log_interval=int(args.log_interval),
            reward_loss_weight=float(args.reward_loss_weight),
        )
        if test_loader is not None
        else None
    )
    ckpt_path = save_policy_checkpoint(
        args.out,
        model,
        stats=stats,
        image_size=int(args.image_size),
        model_type=args.model_type,
        meta={
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_type": args.model_type,
            "optimizer": args.optimizer,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_epochs": float(args.warmup_epochs),
            "warmup_steps": int(warmup_steps),
            "lookahead_frames": int(args.lookahead_frames),
            "action_normalization": str(args.action_normalization),
            "action_scale_deg": float(args.action_scale_deg),
            "gripper_action_scale": float(args.gripper_action_scale),
            "target_feature_std_floor": float(args.target_feature_std_floor),
            "state_clip": float(args.state_clip),
            "cache_images": bool(args.cache_images),
            "preload_images": bool(args.preload_images),
            "load_images": bool(load_images),
            "amp": bool(args.amp and device.type == "cuda"),
            "grad_clip_norm": float(args.grad_clip_norm),
            "early_stopping_patience": int(args.early_stopping_patience),
            "best_epoch": int(best_epoch),
            "episodes": [path.as_posix() for path in episode_dirs],
            "train_episodes": [path.as_posix() for path in train_episode_dirs],
            "val_episodes": [path.as_posix() for path in val_episode_dirs],
            "test_episodes": [path.as_posix() for path in test_episode_dirs],
            "split_by_episode": bool(args.split_by_episode),
            "target_conditioning": bool(args.target_conditioning),
            "reward_loss_weight": float(args.reward_loss_weight),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "best_val_loss": best_val,
            "final_test_metrics": final_test_metrics,
            "history": history,
        },
    )
    history_path = ckpt_path.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved policy checkpoint: {ckpt_path}")
    print(f"Saved training history: {history_path}")


if __name__ == "__main__":
    main()
