#!/usr/bin/env python3
"""End-to-end experiment runner for the plan shown in the design image.

The workflow is:
1. Baseline CV across all candidate models.
2. Select the top 2 models automatically from the latest CV results.
3. Run the acceleration sweep for those 2 models.
4. Select the best acceleration / best model.
5. Train the winning model longer until convergence.
6. Hyperparameter-tune the finalists.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_BASE_DIR = os.path.expanduser("~/scratch/MRI_DATASET/Nelson_runs")
DEFAULT_MODELS = ["CascadeNet", "UNet", "DUNDD", "MoDL", "E2EVarNet"]


def run_cmd(command):
    print(f"\n>>> {' '.join(command)}")
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def latest_run_id(model_name):
    run_dir = Path(RUNS_BASE_DIR)
    if not run_dir.exists():
        raise FileNotFoundError(f"No runs found in {run_dir}")
    matches = sorted(p.name for p in run_dir.iterdir() if p.is_dir() and p.name.startswith(f"{model_name}_"))
    if not matches:
        raise FileNotFoundError(f"No runs found for model '{model_name}' in {run_dir}")
    return matches[-1]


def cv_summary_for_model(model_name):
    run_id = latest_run_id(model_name)
    result_path = Path(RUNS_BASE_DIR) / run_id / "results" / f"cv_results_{model_name}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing CV results at {result_path}")
    with open(result_path, "r") as f:
        results = json.load(f)
    scores = [float(v["best_ssim"]) for v in results.values()]
    mean_ssim = sum(scores) / len(scores) if scores else 0.0
    return {"model": model_name, "run_id": run_id, "mean_ssim": mean_ssim, "scores": scores}


def choose_top_models(models, top_n=2):
    summaries = [cv_summary_for_model(m) for m in models]
    summaries = sorted(summaries, key=lambda s: s["mean_ssim"], reverse=True)
    return summaries[:top_n]


def best_acceleration_for_model(model_name, run_id):
    result_path = Path(RUNS_BASE_DIR) / run_id / "results" / f"reliability_sweep_{model_name}.json"
    if not result_path.exists():
        return 2
    with open(result_path, "r") as f:
        sweep = json.load(f)
    best_r = 2
    best_mean = -1e9
    for r, payload in sweep.items():
        if not isinstance(payload, dict):
            continue
        model_ssim = payload.get("model_ssim", [])
        if not model_ssim:
            continue
        m = sum(model_ssim) / len(model_ssim)
        if m > best_mean:
            best_mean = m
            best_r = int(r)
    return best_r


def run_cv(model_name, epochs, batch_size, acceleration=2):
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "main_cv.py"),
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--acceleration", str(acceleration),
    ]
    run_cmd(cmd)
    return latest_run_id(model_name)


def run_sweep(model_name, run_id):
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "evaluate_2.py"),
        "--task", "sweep",
        "--model", model_name,
        "--run_id", run_id,
    ]
    run_cmd(cmd)


def run_tune(model_name, epochs, batch_size, acceleration, tuning_config):
    for key, value in tuning_config.items():
        pass
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "main_cv.py"),
        "--model", model_name,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--acceleration", str(acceleration),
    ]
    run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run the project experiment plan from the image.")
    parser.add_argument("--phase", choices=[
        "baseline_cv",
        "top2",
        "acceleration_sweep",
        "extended_training",
        "hyperparameter_tuning",
        "all",
    ], default="all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--acceleration", type=int, default=2)
    parser.add_argument("--extended_epochs", type=int, default=300)
    parser.add_argument("--top_n", type=int, default=2)
    args = parser.parse_args()

    selected = [args.phase] if args.phase != "all" else [
        "baseline_cv",
        "top2",
        "acceleration_sweep",
        "extended_training",
        "hyperparameter_tuning",
    ]

    if "baseline_cv" in selected:
        for model in DEFAULT_MODELS:
            run_cv(model, epochs=args.epochs, batch_size=args.batch_size, acceleration=args.acceleration)

    if "top2" in selected:
        top_models = choose_top_models(DEFAULT_MODELS, top_n=args.top_n)
        print("\nTop 2 models selected by CV mean SSIM:")
        for item in top_models:
            print(f"  - {item['model']}: mean SSIM = {item['mean_ssim']:.4f} (run={item['run_id']})")

    if "acceleration_sweep" in selected:
        top_models = choose_top_models(DEFAULT_MODELS, top_n=args.top_n)
        for item in top_models:
            model = item["model"]
            run_id = item["run_id"]
            run_sweep(model, run_id)
            best_r = best_acceleration_for_model(model, run_id)
            print(f"\nBest acceleration for {model}: R={best_r}")

    if "extended_training" in selected:
        top_models = choose_top_models(DEFAULT_MODELS, top_n=args.top_n)
        winner = top_models[0]["model"]
        winner_run = top_models[0]["run_id"]
        best_r = best_acceleration_for_model(winner, winner_run)
        run_cv(winner, epochs=args.extended_epochs, batch_size=args.batch_size, acceleration=best_r)
        print(f"\nExtended training completed for {winner} at R={best_r} for {args.extended_epochs} epochs")

    if "hyperparameter_tuning" in selected:
        top_models = choose_top_models(DEFAULT_MODELS, top_n=args.top_n)
        for item in top_models:
            model = item["model"]
            run_id = item["run_id"]
            best_r = best_acceleration_for_model(model, run_id)
            tuning_configs = [
                {"batch_size": 8, "epochs": args.epochs},
                {"batch_size": 16, "epochs": args.epochs},
                {"batch_size": 8, "epochs": min(args.epochs + 10, 90)},
            ]
            for cfg in tuning_configs:
                run_tune(model, epochs=cfg["epochs"], batch_size=cfg["batch_size"], acceleration=best_r, tuning_config=cfg)

    print("\nExperiment plan complete.")


if __name__ == "__main__":
    main()
