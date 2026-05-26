from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from fr5_il_dataset import DEFAULT_POLICY_DIR, resolve_demo_path
from fr5_policy_rollout import DEFAULT_ROLLOUT_LOG_DIR


def discover_policies(policy_dir: str | Path, pattern: str) -> list[Path]:
    root = resolve_demo_path(policy_dir)
    policies = sorted(root.glob(pattern))
    if not policies:
        raise RuntimeError(f"No policies found under {root} with pattern {pattern!r}")
    return policies


def load_rollout_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_row(meta: dict, *, policy: Path, condition: str, seed: int, json_path: Path) -> dict:
    summary = meta.get("summary", {})
    per_tape = summary.get("per_tape_results", {}) if isinstance(summary.get("per_tape_results", {}), dict) else {}
    tape_success_count = sum(1 for item in per_tape.values() if bool(item.get("place_success", False)))
    sim_rgb_source = str(summary.get("sim_rgb_source", ""))
    training_rgb_source = str(summary.get("training_rgb_source", ""))
    rgb_source_valid = not ("visual_render" in training_rgb_source and sim_rgb_source != "visual")
    return {
        "policy_name": policy.stem,
        "policy": policy.as_posix(),
        "condition": condition,
        "sim_seed": int(seed),
        "task_success": bool(summary.get("task_success", False)),
        "tape_success_count": int(tape_success_count),
        "tape_count": int(len(per_tape)),
        "place_error_xy_m": float(summary.get("place_error_xy_m", np.nan)),
        "place_success_radius_m": float(summary.get("place_success_radius_m", np.nan)),
        "frames": int(summary.get("frames", 0)),
        "clip_fraction": float(summary.get("clip_fraction", np.nan)),
        "min_tcp_z_m": float(summary.get("min_tcp_z_m", np.nan)),
        "max_tcp_step_m": float(summary.get("max_tcp_step_m", np.nan)),
        "sim_rgb_source": sim_rgb_source,
        "training_rgb_source": training_rgb_source,
        "rgb_source_valid": bool(rgb_source_valid),
        "attach_ever": bool(summary.get("attach_ever", False)),
        "attach_first_frame": int(summary.get("attach_first_frame", -1)),
        "min_tcp_object_dist_xy_m": float(summary.get("min_tcp_object_dist_xy_m", np.nan)),
        "final_tcp_object_dist_xy_m": float(summary.get("final_tcp_object_dist_xy_m", np.nan)),
        "min_tcp_goal_dist_xy_m": float(summary.get("min_tcp_goal_dist_xy_m", np.nan)),
        "min_object_goal_error_xy_m": float(summary.get("min_object_goal_error_xy_m", np.nan)),
        "object_moved_xy_m": float(summary.get("object_moved_xy_m", np.nan)),
        "attach_assist": bool(summary.get("attach_assist", condition == "assist")),
        "phase_guard": str(summary.get("phase_guard", "off")),
        "phase_guard_active": bool(summary.get("phase_guard_active", False)),
        "phase_guard_final_phase": str(summary.get("phase_guard_final_phase", "")),
        "phase_guard_residual_scale": float(summary.get("phase_guard_residual_scale", 0.0)),
        "phase_guard_soft_gain": float(summary.get("phase_guard_soft_gain", 0.0)),
        "policy_ablation": str(summary.get("policy_ablation", "normal")),
        "rollout_json": json_path.as_posix(),
        "rollout_npz": json_path.with_suffix(".npz").as_posix(),
    }


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["policy_name"]),
                str(row["condition"]),
                str(row.get("phase_guard", "off")),
                str(row.get("policy_ablation", "normal")),
            ),
            [],
        ).append(row)
    out = []
    for (policy_name, condition, phase_guard, policy_ablation), group in sorted(groups.items()):
        success = np.asarray([float(item["task_success"]) for item in group], dtype=np.float64)
        place = np.asarray([float(item["place_error_xy_m"]) for item in group], dtype=np.float64)
        clip = np.asarray([float(item["clip_fraction"]) for item in group], dtype=np.float64)
        tcp_z = np.asarray([float(item["min_tcp_z_m"]) for item in group], dtype=np.float64)
        attach = np.asarray([float(item.get("attach_ever", False)) for item in group], dtype=np.float64)
        approach = np.asarray([float(item.get("min_tcp_object_dist_xy_m", np.nan)) for item in group], dtype=np.float64)
        moved = np.asarray([float(item.get("object_moved_xy_m", np.nan)) for item in group], dtype=np.float64)
        rgb_valid = np.asarray([float(item.get("rgb_source_valid", False)) for item in group], dtype=np.float64)
        tape_counts = np.asarray([float(item.get("tape_success_count", np.nan)) for item in group], dtype=np.float64)
        out.append(
            {
                "policy_name": policy_name,
                "condition": condition,
                "phase_guard": phase_guard,
                "policy_ablation": policy_ablation,
                "episodes": int(len(group)),
                "success_rate": float(np.mean(success)) if success.size else 0.0,
                "success_count": int(np.sum(success)),
                "tape_success_count_mean": float(np.nanmean(tape_counts)),
                "place_error_mean_m": float(np.nanmean(place)),
                "place_error_median_m": float(np.nanmedian(place)),
                "place_error_std_m": float(np.nanstd(place)),
                "clip_fraction_mean": float(np.nanmean(clip)),
                "min_tcp_z_mean_m": float(np.nanmean(tcp_z)),
                "attach_rate": float(np.nanmean(attach)) if attach.size else 0.0,
                "min_tcp_object_dist_mean_m": float(np.nanmean(approach)),
                "object_moved_mean_m": float(np.nanmean(moved)),
                "rgb_source_valid_rate": float(np.nanmean(rgb_valid)) if rgb_valid.size else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate all FR5 BC policies in randomized simulated task rollouts.")
    parser.add_argument("--policy-dir", type=str, default=DEFAULT_POLICY_DIR.as_posix())
    parser.add_argument("--policy-pattern", type=str, default="fr5_bc_*_tape_bc_v1.pt")
    parser.add_argument("--policies", nargs="*", default=None, help="Explicit policy paths. Overrides --policy-dir/--policy-pattern.")
    parser.add_argument("--out-dir", type=str, default=(DEFAULT_ROLLOUT_LOG_DIR / "batch_eval").as_posix())
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260515)
    parser.add_argument("--conditions", nargs="+", choices=["assist", "noassist"], default=["assist", "noassist"])
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--lookahead-frames", type=int, default=0, help="0 uses each policy checkpoint value.")
    parser.add_argument("--max-action-deg", type=float, default=15.0)
    parser.add_argument("--max-gripper-delta", type=float, default=0.05)
    parser.add_argument(
        "--phase-guard",
        choices=["off", "soft", "scripted", "inference"],
        default="off",
        help="Single validation tier. inference is a legacy alias for scripted.",
    )
    parser.add_argument(
        "--phase-guards",
        nargs="*",
        choices=["off", "soft", "scripted", "inference"],
        default=None,
        help="Run several validation tiers in one batch, e.g. --phase-guards off soft scripted.",
    )
    parser.add_argument("--phase-guard-residual-scale", type=float, default=0.0)
    parser.add_argument("--phase-guard-residual-max-deg", type=float, default=2.0)
    parser.add_argument("--phase-guard-soft-gain", type=float, default=0.35)
    parser.add_argument("--phase-guard-soft-max-deg", type=float, default=6.0)
    parser.add_argument("--phase-guard-xy-tol", type=float, default=0.025)
    parser.add_argument("--phase-guard-z-tol", type=float, default=0.04)
    parser.add_argument("--phase-guard-close-frames", type=int, default=8)
    parser.add_argument("--phase-guard-release-frames", type=int, default=5)
    parser.add_argument("--phase-guard-max-close-frames", type=int, default=30)
    parser.add_argument("--phase-guard-max-release-frames", type=int, default=30)
    parser.add_argument("--phase-guard-max-phase-frames", type=int, default=80)
    parser.add_argument("--phase-guard-ik-iters", type=int, default=50)
    parser.add_argument("--action-ema", type=float, default=0.0)
    parser.add_argument("--gripper-ema", type=float, default=0.0)
    parser.add_argument("--max-action-step-deg", type=float, default=0.0)
    parser.add_argument(
        "--policy-ablation",
        choices=["normal", "zero_image", "zero_target", "zero_policy", "random_policy"],
        default="normal",
        help="Single ablation mode.",
    )
    parser.add_argument(
        "--policy-ablations",
        nargs="*",
        choices=["normal", "zero_image", "zero_target", "zero_policy", "random_policy"],
        default=None,
        help="Run several policy ablations in one batch.",
    )
    parser.add_argument("--policy-ablation-seed", type=int, default=91017)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sim-rgb-source", choices=["auto", "visual", "camera-solid"], default="auto")
    parser.add_argument("--allow-rgb-source-mismatch", action="store_true")
    parser.add_argument(
        "--no-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --no-no-window to keep RenderApp windows open. visual RGB source automatically requires a window.",
    )
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    policies = [resolve_demo_path(p) for p in args.policies] if args.policies else discover_policies(args.policy_dir, args.policy_pattern)
    seeds = list(args.seeds) if args.seeds else [int(args.base_seed) + idx for idx in range(int(args.num_seeds))]
    phase_guards = list(args.phase_guards) if args.phase_guards else [str(args.phase_guard)]
    policy_ablations = list(args.policy_ablations) if args.policy_ablations else [str(args.policy_ablation)]
    run_name = args.run_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    formal_run = any(token in run_name.lower() for token in ("thesis", "paper", "final"))
    if formal_run and (bool(args.allow_rgb_source_mismatch) or str(args.sim_rgb_source) == "camera-solid"):
        raise RuntimeError(
            f"Refusing formal run_name={run_name!r} with debug RGB settings. "
            "Use --sim-rgb-source visual for thesis metrics, or rename the run to debug_*."
        )
    out_dir = resolve_demo_path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_script = Path(__file__).resolve().parent / "fr5_policy_rollout.py"
    rows: list[dict] = []
    failures: list[dict] = []

    print("Batch policy evaluation:")
    print(f"  policies={len(policies)}")
    print(f"  seeds={seeds}")
    print(f"  conditions={args.conditions}")
    print(f"  phase_guards={phase_guards}")
    print(f"  policy_ablations={policy_ablations}")
    print(f"  out_dir={out_dir}")

    for policy in policies:
        for condition in args.conditions:
            for phase_guard in phase_guards:
                for policy_ablation in policy_ablations:
                    for seed in seeds:
                        phase_guard_name = "scripted" if str(phase_guard) == "inference" else str(phase_guard)
                        suffix = f"{policy.stem}_{condition}_{phase_guard_name}_{policy_ablation}_seed{int(seed)}"
                        cmd = [
                            sys.executable,
                            rollout_script.as_posix(),
                            "--policy",
                            policy.as_posix(),
                            "--sim-task-rollout",
                            "--sim-seed",
                            str(int(seed)),
                            "--max-steps",
                            str(int(args.max_steps)),
                            "--hz",
                            str(float(args.hz)),
                            "--lookahead-frames",
                            str(int(args.lookahead_frames)),
                            "--max-action-deg",
                            str(float(args.max_action_deg)),
                            "--max-gripper-delta",
                            str(float(args.max_gripper_delta)),
                            "--phase-guard",
                            phase_guard,
                            "--phase-guard-residual-scale",
                            str(float(args.phase_guard_residual_scale)),
                            "--phase-guard-residual-max-deg",
                            str(float(args.phase_guard_residual_max_deg)),
                            "--phase-guard-soft-gain",
                            str(float(args.phase_guard_soft_gain)),
                            "--phase-guard-soft-max-deg",
                            str(float(args.phase_guard_soft_max_deg)),
                            "--phase-guard-xy-tol",
                            str(float(args.phase_guard_xy_tol)),
                            "--phase-guard-z-tol",
                            str(float(args.phase_guard_z_tol)),
                            "--phase-guard-close-frames",
                            str(int(args.phase_guard_close_frames)),
                            "--phase-guard-release-frames",
                            str(int(args.phase_guard_release_frames)),
                            "--phase-guard-max-close-frames",
                            str(int(args.phase_guard_max_close_frames)),
                            "--phase-guard-max-release-frames",
                            str(int(args.phase_guard_max_release_frames)),
                            "--phase-guard-max-phase-frames",
                            str(int(args.phase_guard_max_phase_frames)),
                            "--phase-guard-ik-iters",
                            str(int(args.phase_guard_ik_iters)),
                            "--action-ema",
                            str(float(args.action_ema)),
                            "--gripper-ema",
                            str(float(args.gripper_ema)),
                            "--max-action-step-deg",
                            str(float(args.max_action_step_deg)),
                            "--policy-ablation",
                            policy_ablation,
                            "--policy-ablation-seed",
                            str(int(args.policy_ablation_seed)),
                            "--device",
                            args.device,
                            "--log-dir",
                            out_dir.as_posix(),
                            "--log-suffix",
                            suffix,
                            "--log-every",
                            "0",
                            "--no-diagnose-control",
                            "--sim-rgb-source",
                            args.sim_rgb_source,
                        ]
                        if args.allow_rgb_source_mismatch:
                            cmd.append("--allow-rgb-source-mismatch")
                        effective_no_window = bool(args.no_window) and args.sim_rgb_source == "camera-solid"
                        if effective_no_window:
                            cmd.append("--no-window")
                        cmd.append("--sim-attach-assist" if condition == "assist" else "--no-sim-attach-assist")
                        print(
                            f"\n=== {policy.stem} | {condition} | phase_guard={phase_guard_name} "
                            f"| ablation={policy_ablation} | seed={seed} ===",
                            flush=True,
                        )
                        run_log = out_dir / f"run_{suffix}.log"
                        with run_log.open("w", encoding="utf-8") as log_f:
                            log_f.write(" ".join(str(item) for item in cmd) + "\n\n")
                            log_f.flush()
                            proc = subprocess.run(
                                cmd,
                                cwd=Path(__file__).resolve().parents[2].as_posix(),
                                check=False,
                                stdout=log_f,
                                stderr=subprocess.STDOUT,
                                text=True,
                            )
                        if proc.returncode != 0:
                            tail = "\n".join(run_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
                            failure = {
                                "policy": policy.as_posix(),
                                "condition": condition,
                                "phase_guard": phase_guard_name,
                                "policy_ablation": policy_ablation,
                                "seed": int(seed),
                                "returncode": int(proc.returncode),
                                "run_log": run_log.as_posix(),
                                "error_tail": tail,
                            }
                            failures.append(failure)
                            print(f"FAILED: {failure}", flush=True)
                            if not args.keep_going:
                                raise RuntimeError(f"Rollout failed: {failure}")
                            continue
                        matches = sorted(out_dir.glob(f"simtask_{policy.stem}_*_{suffix}*.json"))
                        if not matches:
                            failure = {
                                "policy": policy.as_posix(),
                                "condition": condition,
                                "phase_guard": phase_guard_name,
                                "policy_ablation": policy_ablation,
                                "seed": int(seed),
                                "returncode": 0,
                                "error": "missing rollout json",
                            }
                            failures.append(failure)
                            print(f"FAILED: {failure}", flush=True)
                            if not args.keep_going:
                                raise RuntimeError(f"Rollout JSON not found for {suffix}")
                            continue
                        json_path = matches[-1]
                        rows.append(
                            extract_row(load_rollout_json(json_path), policy=policy, condition=condition, seed=int(seed), json_path=json_path)
                        )

    aggregates = aggregate_rows(rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "policies": [p.as_posix() for p in policies],
        "seeds": [int(s) for s in seeds],
        "conditions": list(args.conditions),
        "phase_guards": ["scripted" if str(item) == "inference" else str(item) for item in phase_guards],
        "policy_ablations": list(policy_ablations),
        "max_steps": int(args.max_steps),
        "hz": float(args.hz),
        "max_action_deg": float(args.max_action_deg),
        "max_gripper_delta": float(args.max_gripper_delta),
        "phase_guard": str(args.phase_guard),
        "phase_guards_requested": list(args.phase_guards) if args.phase_guards else None,
        "phase_guard_residual_scale": float(args.phase_guard_residual_scale),
        "phase_guard_residual_max_deg": float(args.phase_guard_residual_max_deg),
        "phase_guard_soft_gain": float(args.phase_guard_soft_gain),
        "phase_guard_soft_max_deg": float(args.phase_guard_soft_max_deg),
        "phase_guard_xy_tol": float(args.phase_guard_xy_tol),
        "phase_guard_z_tol": float(args.phase_guard_z_tol),
        "phase_guard_close_frames": int(args.phase_guard_close_frames),
        "phase_guard_release_frames": int(args.phase_guard_release_frames),
        "phase_guard_max_close_frames": int(args.phase_guard_max_close_frames),
        "phase_guard_max_release_frames": int(args.phase_guard_max_release_frames),
        "phase_guard_max_phase_frames": int(args.phase_guard_max_phase_frames),
        "phase_guard_ik_iters": int(args.phase_guard_ik_iters),
        "action_ema": float(args.action_ema),
        "gripper_ema": float(args.gripper_ema),
        "max_action_step_deg": float(args.max_action_step_deg),
        "policy_ablation": str(args.policy_ablation),
        "policy_ablations_requested": list(args.policy_ablations) if args.policy_ablations else None,
        "policy_ablation_seed": int(args.policy_ablation_seed),
        "sim_rgb_source": str(args.sim_rgb_source),
        "allow_rgb_source_mismatch": bool(args.allow_rgb_source_mismatch),
        "no_window": bool(args.no_window),
        "effective_no_window": bool(args.no_window) and str(args.sim_rgb_source) == "camera-solid",
        "formal_run_guard": bool(formal_run),
        "valid_for_thesis": bool(not args.allow_rgb_source_mismatch and str(args.sim_rgb_source) != "camera-solid"),
        "rows": rows,
        "aggregates": aggregates,
        "failures": failures,
    }
    summary_path = out_dir / "policy_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "policy_eval_rows.csv", rows)
    write_csv(out_dir / "policy_eval_aggregates.csv", aggregates)

    print("\nAggregate results:")
    for row in aggregates:
        print(
            f"  {row['policy_name']} | {row['condition']}: "
            f"phase_guard={row.get('phase_guard', 'off')}, "
            f"ablation={row.get('policy_ablation', 'normal')}, "
            f"success={row['success_rate']:.3f} ({row['success_count']}/{row['episodes']}), "
            f"tapes={row.get('tape_success_count_mean', 0.0):.2f}, "
            f"place_err_mean={row['place_error_mean_m']:.4f}m",
            flush=True,
        )
    print(f"Saved summary: {summary_path}")
    print(f"Saved CSV: {out_dir / 'policy_eval_rows.csv'}")
    print(f"Saved CSV: {out_dir / 'policy_eval_aggregates.csv'}")


if __name__ == "__main__":
    main()
