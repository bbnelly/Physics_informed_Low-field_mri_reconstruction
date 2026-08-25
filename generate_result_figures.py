"""Generate training-loss and best-fold acceleration figures from saved runs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import load_and_separate_dataset, ValMRIDataset
from model_registry import model_factories
from evaluate_2 import get_cv_results, load_fold_checkpoint, per_slice_metrics, kspace_to_image
from run_manager import load_run
from masks import create_ky_kz_undersampling_mask


DEFAULT_MODELS = ("CascadeNet", "DUNDD", "E2EVarNet", "MoDL", "UNet")
ACCELERATIONS = (2, 4, 6, 8, 10, 12)
COLORS = ("#176b87", "#e07a5f", "#3a7d44", "#8e5ea2", "#c28f2c")


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def latest_run(runs_dir, model_name):
    candidates = sorted(runs_dir.glob(f"{model_name}_*/"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def completed_histories(run_dir, model_name):
    histories = []
    for path in sorted((run_dir / "checkpoints").glob(f"history_{model_name}_fold*.json")):
        history = load_json(path)
        if history.get("train_loss"):
            histories.append((path.stem, history))
    return histories


def plot_loss_curves(run_dir, model_name, output_dir):
    histories = completed_histories(run_dir, model_name)
    if not histories:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, (label, history) in enumerate(histories):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], linewidth=1.8,
                color=COLORS[index % len(COLORS)], label=label.replace("history_", ""))
    ax.set_title(f"Training Loss by Fold - {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = output_dir / f"loss_curves_{model_name}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def best_fold(run_dir, model_name):
    results_path = run_dir / "results" / f"cv_results_{model_name}.json"
    results = load_json(results_path)
    subject, details = max(results.items(), key=lambda item: item[1]["best_ssim"])
    return details["fold"], subject, details


def plot_acceleration_comparison(run_dirs, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    plotted = 0
    for index, (model_name, run_dir) in enumerate(run_dirs.items()):
        try:
            fold, subject, details = best_fold(run_dir, model_name)
            sweep = load_json(run_dir / "results" / f"reliability_sweep_{model_name}.json")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue

        psnr_values = [sweep[str(acceleration)]["model_psnr"][fold - 1] for acceleration in ACCELERATIONS]
        ssim_values = [sweep[str(acceleration)]["model_ssim"][fold - 1] for acceleration in ACCELERATIONS]
        color = COLORS[index % len(COLORS)]
        label = f"{model_name} (fold {fold}, val {subject})"
        axes[0].plot(ACCELERATIONS, psnr_values, marker="o", color=color, label=label)
        axes[1].plot(ACCELERATIONS, ssim_values, marker="o", color=color, label=label)
        for acceleration, psnr, ssim in zip(ACCELERATIONS, psnr_values, ssim_values):
            axes[0].annotate(f"{psnr:.2f}", (acceleration, psnr), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=7, color=color)
            axes[1].annotate(f"{ssim:.3f}", (acceleration, ssim), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=7, color=color)
        plotted += 1

    if not plotted:
        plt.close(figure)
        return None
    for ax, title, ylabel in zip(axes, ("PSNR by Acceleration", "SSIM by Acceleration"),
                                 ("PSNR (dB)", "SSIM")):
        ax.set_title(title)
        ax.set_xlabel("Acceleration factor (R)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ACCELERATIONS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    figure.suptitle("Best-Fold Quality Across Acceleration Factors", fontweight="bold")
    figure.tight_layout()
    path = output_dir / "best_fold_acceleration_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def target_normalized(image, target):
    target_min = target.min()
    target_range = target.max() - target_min + 1e-8
    return np.clip((image - target_min) / target_range, 0, 1)


def plot_reconstruction_slices(run_dirs, output_dir, device):
    """Save one clean six-panel middle-slice figure per model."""
    _, fully_sampled_df, _ = load_and_separate_dataset()
    generated_paths = []
    for model_name, run_dir in run_dirs.items():
        try:
            run = load_run(run_dir.name, base_dir=run_dir.parent)
            results = get_cv_results(run, model_name)
            subject, details = max(results.items(), key=lambda item: item[1]["best_ssim"])
            model, _ = load_fold_checkpoint(run, model_name, details["fold"], subject, device)
            sweep = load_json(run_dir / "results" / f"reliability_sweep_{model_name}.json")
            fold_val_df = fully_sampled_df[fully_sampled_df["subject"] == subject].reset_index(drop=True)
            val_set = ValMRIDataset(fold_val_df, acceleration=ACCELERATIONS[0])
            volume = val_set.volumes[0]
            slice_idx = volume.shape[0] // 2
            target = torch.from_numpy(np.stack([volume.real, volume.imag])).float()
        except (FileNotFoundError, KeyError, IndexError, json.JSONDecodeError):
            continue
        target_vol = kspace_to_image(target.numpy())
        target_img = target_vol[slice_idx]
        figure, axes = plt.subplots(2, 3, figsize=(10, 7), squeeze=False)
        for col_idx, acceleration in enumerate(ACCELERATIONS):
            seed = 1000 + acceleration
            mask = torch.from_numpy(create_ky_kz_undersampling_mask(
                volume.shape[0], volume.shape[1], volume.shape[2],
                acceleration=acceleration, seed=seed,
            )).float()
            undersampled = volume * mask.numpy()
            inp = torch.from_numpy(np.stack([undersampled.real, undersampled.imag])).float()
            with torch.no_grad():
                output = model(inp.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
            fold_psnr = sweep[str(acceleration)]["model_psnr"][details["fold"] - 1]
            fold_ssim = sweep[str(acceleration)]["model_ssim"][details["fold"] - 1]
            output_vol = kspace_to_image(output[0].cpu().numpy())
            output_img = target_normalized(output_vol[slice_idx], target_img)
            row_idx, panel_idx = divmod(col_idx, 3)
            ax = axes[row_idx, panel_idx]
            ax.imshow(output_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"R={acceleration}\nFold PSNR {fold_psnr:.2f} dB\nFold SSIM {fold_ssim:.3f}", fontsize=10)
            ax.axis("off")
        figure.suptitle(f"{model_name} - Middle Held-out Brain Slice Across Acceleration Factors\n"
                        f"Best fold {details['fold']} | Subject {subject} | Middle slice {slice_idx}",
                        fontsize=13, fontweight="bold")
        figure.tight_layout()
        path = output_dir / f"best_fold_acceleration_slices_{model_name}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        generated_paths.append(path)
    return generated_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    output_dir = args.output_dir or args.runs_dir / "analysis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {}
    for model_name in args.models:
        run_dir = latest_run(args.runs_dir, model_name)
        if run_dir is None:
            print(f"Skipping {model_name}: no run directory")
            continue
        loss_path = plot_loss_curves(run_dir, model_name, output_dir)
        if loss_path:
            print(f"Saved {loss_path}")
        if (run_dir / "results" / f"reliability_sweep_{model_name}.json").exists():
            run_dirs[model_name] = run_dir
    acceleration_path = plot_acceleration_comparison(run_dirs, output_dir)
    if acceleration_path:
        print(f"Saved {acceleration_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    slices_path = plot_reconstruction_slices(run_dirs, output_dir, device)
    for path in slices_path or []:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()