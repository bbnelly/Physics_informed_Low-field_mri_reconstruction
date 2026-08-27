"""Generate training-loss and best-fold acceleration figures from saved runs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data_loader import load_and_separate_dataset, ValMRIDataset
from model_registry import model_factories
from evaluate_2 import get_cv_results, load_fold_checkpoint, per_slice_metrics, kspace_to_image
from run_manager import load_run
from masks import create_ky_kz_undersampling_mask


DEFAULT_MODELS = ("CascadeNet", "DUNDD", "E2EVarNet", "MoDL", "UNet")
DEFAULT_RUN_DIRS = {
    "CascadeNet": Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs/CascadeNet_figures_fix_20260825_v2_20260825-132715_job1691089"),
    "DUNDD": Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs/DUNDD_figures_fix_20260825_v2_20260825-132715_job1691088"),
    "E2EVarNet": Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs/E2EVarNet_figures_fix_20260825_v2_20260825-132715_job1691091"),
    "MoDL": Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs/MoDL_figures_fix_20260825_v2_20260825-132714_job1691090"),
    "UNet": Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs/UNet_20260825-072546_job1659489"),
}
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
    line_styles = ("-", "--", "-.", ":", (0, (5, 1)))
    markers = ("o", "s", "^", "D", "P")
    figure, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True)
    plotted = 0
    for index, (model_name, run_dir) in enumerate(run_dirs.items()):
        try:
            fold, subject, details = best_fold(run_dir, model_name)
            sweep = load_json(run_dir / "results" / f"reliability_sweep_{model_name}.json")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            continue

        psnr_values = [sweep[str(acceleration)]["model_psnr"][fold - 1] for acceleration in ACCELERATIONS]
        ssim_values = [sweep[str(acceleration)]["model_ssim"][fold - 1] for acceleration in ACCELERATIONS]
        label = f"{model_name} (fold {fold}, val {subject})"
        style = {"color": "black", "linestyle": line_styles[index],
                 "marker": markers[index], "linewidth": 2.0, "markersize": 7,
                 "markerfacecolor": "white", "markeredgewidth": 1.4,
                 "label": label}
        axes[0].plot(ACCELERATIONS, psnr_values, **style)
        axes[1].plot(ACCELERATIONS, ssim_values, **style)
        plotted += 1

    if not plotted:
        plt.close(figure)
        return None
    for ax, title, ylabel in zip(axes, ("Peak Signal-to-Noise Ratio", "Structural Similarity"),
                                 ("PSNR (dB)", "SSIM")):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Acceleration factor, R", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(ACCELERATIONS)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, color="0.82", linestyle=":", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, loc="best", frameon=True, facecolor="white", edgecolor="black")
    figure.suptitle("Five-Model Acceleration Sweep Using Best Validation Folds",
                    fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "acceleration_sweep_5_models_black_white.png"
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def target_normalized(image, target):
    target_min = target.min()
    target_range = target.max() - target_min + 1e-8
    return np.clip((image - target_min) / target_range, 0, 1)


def slice_metrics(pred, target):
    target_min = target.min()
    target_range = target.max() - target_min + 1e-8
    pred_scaled = np.clip((pred - target_min) / target_range, 0, 1)
    target_scaled = np.clip((target - target_min) / target_range, 0, 1)
    return (
        float(peak_signal_noise_ratio(target_scaled, pred_scaled, data_range=1)),
        float(structural_similarity(target_scaled, pred_scaled, data_range=1)),
    )


def plot_reconstruction_slices(run_dirs, output_dir, device, acceleration=2, slice_idx=27):
    """Save a black-and-white 4x5 comparison at one central volume slice."""
    _, fully_sampled_df, _ = load_and_separate_dataset()
    panels = {}
    for model_name, run_dir in run_dirs.items():
        run = load_run(run_dir.name, base_dir=run_dir.parent)
        results = get_cv_results(run, model_name)
        subject, details = max(results.items(), key=lambda item: item[1]["best_ssim"])
        model, _ = load_fold_checkpoint(run, model_name, details["fold"], subject, device)
        if model is None:
            raise FileNotFoundError(f"Best checkpoint is missing for {model_name}")
        fold_val_df = fully_sampled_df[fully_sampled_df["subject"] == subject].reset_index(drop=True)
        volume = ValMRIDataset(fold_val_df, acceleration=acceleration).volumes[0]
        if slice_idx >= volume.shape[0]:
            raise IndexError(f"Slice {slice_idx} is outside {model_name}'s volume shape {volume.shape}")
        target = torch.from_numpy(np.stack([volume.real, volume.imag])).float()
        target_img = kspace_to_image(target.numpy())[slice_idx]
        mask = torch.from_numpy(create_ky_kz_undersampling_mask(
            *volume.shape, acceleration=acceleration, seed=1000 + acceleration,
        )).float()
        inp = torch.from_numpy(np.stack([(volume * mask.numpy()).real,
                                          (volume * mask.numpy()).imag])).float()
        with torch.no_grad():
            output = model(inp.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
        input_img = kspace_to_image(inp.numpy())[slice_idx]
        output_img = kspace_to_image(output[0].cpu().numpy())[slice_idx]
        input_psnr, input_ssim = slice_metrics(input_img, target_img)
        output_psnr, output_ssim = slice_metrics(output_img, target_img)
        target_min = target_img.min()
        target_range = target_img.max() - target_min + 1e-8
        scale = lambda image: np.clip((image - target_min) / target_range, 0, 1)
        panels[model_name] = {
            "input": scale(input_img), "output": scale(output_img),
            "target": scale(target_img), "error": np.abs(scale(output_img) - scale(target_img)),
            "subject": subject, "fold": details["fold"],
            "input_metrics": (input_psnr, input_ssim),
            "output_metrics": (output_psnr, output_ssim),
        }

    row_names = ("Input (zero-filled)", "Model output", "Target", "Absolute error")
    figure, axes = plt.subplots(4, len(run_dirs), figsize=(21, 17), squeeze=False)
    for col_idx, model_name in enumerate(run_dirs):
        panel = panels[model_name]
        input_psnr, input_ssim = panel["input_metrics"]
        output_psnr, output_ssim = panel["output_metrics"]
        titles = (
            f"{model_name}\nInput PSNR {input_psnr:.2f} dB | SSIM {input_ssim:.4f}",
            f"{model_name}\nPSNR {output_psnr:.2f} dB | SSIM {output_ssim:.4f}",
            f"{model_name}\nReference",
            f"{model_name}\nMAE map",
        )
        for row_idx, row_name in enumerate(row_names):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel[('input', 'output', 'target', 'error')[row_idx]],
                      cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(titles[row_idx], fontsize=11, pad=8)
            ax.set_xlabel("Readout x", fontsize=9)
            ax.set_ylabel("Phase encode y" if col_idx == 0 else "", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.text(-0.25, 0.5, row_name, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    figure.suptitle(f"Mid-Brain Reconstruction Comparison | Slice {slice_idx} | R={acceleration}\n"
                    "Best validation fold selected independently for each model",
                    fontsize=17, fontweight="bold")
    figure.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.06, wspace=0.12, hspace=0.28)
    path = output_dir / "mid_brain_reconstruction_4x5.png"
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
    return [path]


def plot_acceleration_slices(model_name, run_dir, output_dir, device, slice_idx=27):
    """Save one model's outputs for every acceleration factor in a 3x2 grid."""
    run = load_run(run_dir.name, base_dir=run_dir.parent)
    results = get_cv_results(run, model_name)
    subject, details = max(results.items(), key=lambda item: item[1]["best_ssim"])
    model, _ = load_fold_checkpoint(run, model_name, details["fold"], subject, device)
    if model is None:
        raise FileNotFoundError(f"{model_name} best-fold checkpoint is missing")

    _, fully_sampled_df, _ = load_and_separate_dataset()
    fold_val_df = fully_sampled_df[fully_sampled_df["subject"] == subject].reset_index(drop=True)
    volume = ValMRIDataset(fold_val_df, acceleration=ACCELERATIONS[0]).volumes[0]
    if slice_idx >= volume.shape[0]:
        raise IndexError(f"Slice {slice_idx} is outside the volume shape {volume.shape}")
    target = torch.from_numpy(np.stack([volume.real, volume.imag])).float()
    target_img = kspace_to_image(target.numpy())[slice_idx]
    target_min = target_img.min()
    target_range = target_img.max() - target_min + 1e-8
    scale = lambda image: np.clip((image - target_min) / target_range, 0, 1)

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), squeeze=False)
    for panel_idx, acceleration in enumerate(ACCELERATIONS):
        mask = torch.from_numpy(create_ky_kz_undersampling_mask(
            *volume.shape, acceleration=acceleration, seed=1000 + acceleration,
        )).float()
        undersampled = volume * mask.numpy()
        inp = torch.from_numpy(np.stack([undersampled.real, undersampled.imag])).float()
        with torch.no_grad():
            output = model(inp.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
        output_img = kspace_to_image(output[0].cpu().numpy())[slice_idx]
        display_img = scale(output_img)
        psnr_value, ssim_value = slice_metrics(output_img, target_img)
        row_idx, col_idx = divmod(panel_idx, 3)
        ax = axes[row_idx, col_idx]
        ax.imshow(display_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"R = {acceleration}\nPSNR = {psnr_value:.2f} dB | SSIM = {ssim_value:.4f}",
                     fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Readout x", fontsize=11, fontweight="bold")
        ax.set_ylabel("Phase encode y", fontsize=11, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9)
    figure.suptitle(f"{model_name} Acceleration Sweep | Slice {slice_idx}\n"
                    f"Best fold {details['fold']} | Held-out subject {subject}",
                    fontsize=17, fontweight="bold")
    figure.text(0.5, 0.02, "Displayed image intensity is normalized using the target slice range",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.05, 1, 0.91))
    path = output_dir / f"{model_name.lower()}_acceleration_sweep_slice{slice_idx}_3x2.png"
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def write_acceleration_summary(run_dirs, output_dir):
    """Write one row per model and acceleration factor using the saved sweeps."""
    import csv

    rows = []
    for model_name, run_dir in run_dirs.items():
        fold, subject, details = best_fold(run_dir, model_name)
        sweep = load_json(run_dir / "results" / f"reliability_sweep_{model_name}.json")
        fold_index = fold - 1
        for acceleration in ACCELERATIONS:
            values = sweep[str(acceleration)]
            rows.append({
                "model": model_name,
                "acceleration_factor": acceleration,
                "best_fold": fold,
                "held_out_subject": subject,
                "validation_psnr_db": values["model_psnr"][fold_index],
                "validation_ssim": values["model_ssim"][fold_index],
                "zero_filled_psnr_db": values["zf_psnr"][fold_index],
                "zero_filled_ssim": values["zf_ssim"][fold_index],
                "best_fold_cv_psnr_db": details["best_psnr"],
                "best_fold_cv_ssim": details["best_ssim"],
            })

    fieldnames = list(rows[0])
    csv_path = output_dir / "acceleration_sweep_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "acceleration_sweep_summary.md"
    with markdown_path.open("w") as handle:
        handle.write("# Acceleration Sweep Summary\n\n")
        handle.write("Best validation fold selected independently for each model; metrics are from the saved LOSO sweep results.\n\n")
        handle.write("| Model | R | Best fold | Held-out subject | PSNR (dB) | SSIM | ZF PSNR (dB) | ZF SSIM |\n")
        handle.write("|---|---:|---:|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['model']} | {row['acceleration_factor']} | {row['best_fold']} | "
                f"{row['held_out_subject']} | {row['validation_psnr_db']:.4f} | "
                f"{row['validation_ssim']:.6f} | {row['zero_filled_psnr_db']:.4f} | "
                f"{row['zero_filled_ssim']:.6f} |\n"
            )
    return csv_path, markdown_path


def write_mean_sweep_tables_and_graph(run_dirs, output_dir):
    """Write five-row PSNR/SSIM tables and plot their acceleration columns."""
    import csv

    metrics = {
        "ssim": ("model_ssim", "SSIM", ".6f"),
        "psnr": ("model_psnr", "PSNR (dB)", ".4f"),
    }
    table_rows = {metric: [] for metric in metrics}
    for model_name, run_dir in run_dirs.items():
        _, _, _ = best_fold(run_dir, model_name)
        sweep = load_json(run_dir / "results" / f"reliability_sweep_{model_name}.json")
        fold, _, _ = best_fold(run_dir, model_name)
        fold_index = fold - 1
        for metric, (value_key, _, _) in metrics.items():
            values = [sweep[str(acceleration)][value_key][fold_index]
                      for acceleration in ACCELERATIONS]
            table_rows[metric].append({
                "model": model_name,
                **{f"R{acceleration}": value for acceleration, value in zip(ACCELERATIONS, values)},
                "mean": float(np.mean(values)),
            })

    outputs = []
    for metric, (_, title, number_format) in metrics.items():
        rows = table_rows[metric]
        fieldnames = ["model"] + [f"R{acceleration}" for acceleration in ACCELERATIONS] + ["mean"]
        csv_path = output_dir / f"acceleration_sweep_mean_{metric}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        markdown_path = output_dir / f"acceleration_sweep_mean_{metric}.md"
        with markdown_path.open("w") as handle:
            handle.write(f"# Acceleration Sweep Mean {title}\n\n")
            handle.write("Mean is across the six acceleration factors shown in the table.\n\n")
            handle.write("| Model | " + " | ".join(f"R={acceleration}" for acceleration in ACCELERATIONS) + " | Mean |\n")
            handle.write("|---|" + "---:|" * (len(ACCELERATIONS) + 1) + "\n")
            for row in rows:
                values = [format(row[f"R{acceleration}"], number_format) for acceleration in ACCELERATIONS]
                handle.write(f"| {row['model']} | " + " | ".join(values) + f" | {row['mean']:{number_format}} |\n")
        outputs.extend((csv_path, markdown_path))

    figure, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True)
    line_styles = ("-", "--", "-.", ":", (0, (5, 1)))
    markers = ("o", "s", "^", "D", "P")
    for axis, metric, (_, title, _) in zip(axes, metrics, metrics.values()):
        for index, row in enumerate(table_rows[metric]):
            axis.plot(
                ACCELERATIONS,
                [row[f"R{acceleration}"] for acceleration in ACCELERATIONS],
                color="black", linestyle=line_styles[index], marker=markers[index],
                linewidth=2.0, markersize=7, markerfacecolor="white",
                markeredgewidth=1.3, label=row["model"],
            )
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_xlabel("Acceleration factor, R", fontsize=12, fontweight="bold")
        axis.set_ylabel(title, fontsize=12, fontweight="bold")
        axis.set_xticks(ACCELERATIONS)
        axis.tick_params(axis="both", labelsize=10)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontweight("bold")
        axis.grid(True, color="0.82", linestyle=":", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(fontsize=10, frameon=True, facecolor="white", edgecolor="black")
    figure.suptitle("Five-Model Acceleration Sweep Summary", fontsize=17, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    graph_path = output_dir / "acceleration_sweep_summary_black_white.png"
    figure.savefig(graph_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    outputs.append(graph_path)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/shadow/scratch/MRI_DATASET/Nelson_runs/main results"))
    parser.add_argument("--acceleration", type=int, default=2)
    parser.add_argument("--slice", dest="slice_idx", type=int, default=27)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    output_dir = args.output_dir or args.runs_dir / "analysis_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {}
    for model_name in args.models:
        run_dir = DEFAULT_RUN_DIRS.get(model_name)
        if run_dir is None:
            print(f"Skipping {model_name}: no run directory")
            continue
        loss_path = plot_loss_curves(run_dir, model_name, output_dir)
        if loss_path:
            print(f"Saved {loss_path}")
        run_dirs[model_name] = run_dir
    acceleration_path = plot_acceleration_comparison(run_dirs, output_dir)
    if acceleration_path:
        print(f"Saved {acceleration_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_name, run_dir in run_dirs.items():
        sweep_path = plot_acceleration_slices(model_name, run_dir, output_dir, device, args.slice_idx)
        print(f"Saved {sweep_path}")
    summary_paths = write_acceleration_summary(run_dirs, output_dir)
    for summary_path in summary_paths:
        print(f"Saved {summary_path}")
    slices_path = plot_reconstruction_slices(run_dirs, output_dir, device, args.acceleration, args.slice_idx)
    for path in slices_path or []:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()