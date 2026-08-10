# main_cv.py
import os
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, OUTPUT_DIR
from data_loader import load_and_separate_dataset
from models import CascadeNet, DUNDD, UNetBaseline, MoDL, E2EVarNet
from train import run_training
from visualize import plot_training_curves


# ── ADD THIS BLOCK ──
logging.basicConfig(
    filename=os.path.join(run_dir, 'run.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)
logger.info(f"Starting LOSO CV — model={model_name}, epochs={num_epochs}, batch_size={batch_size}")

manifest_path = os.path.join(CHECKPOINT_DIR, model_name, 'run_manifest.json')
current_config = {'num_epochs': num_epochs, 'batch_size': batch_size, 'acceleration': acceleration}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        prior_config = json.load(f)
    if prior_config != current_config:
        print(f"⚠️  Config changed since last run ({prior_config} → {current_config}). "
              f"Existing checkpoints may be stale — consider a fresh output directory.")
with open(manifest_path, 'w') as f:
    json.dump(current_config, f, indent=2)

# ── Model factory ──────────────────────────────────────────
model_factories = {
    'UNet': lambda: UNetBaseline(features=32),
    'CascadeNet': lambda: CascadeNet(num_cascades=5, features=32),
    'DUNDD': lambda: DUNDD(num_iterations=5, lambda_dc=0.5, num_channels=64),
    'MoDL': lambda: MoDL(num_iterations=8, num_cg_steps=6, lambda_reg=0.05),
    'E2EVarNet': lambda: E2EVarNet(num_cascades=8, features=32),
}


def run_cross_validation(model_name='CascadeNet', num_epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    """
    Run Leave-One-Subject-Out Cross-Validation.
    Each subject takes turns being the validation set.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"LOSO CROSS-VALIDATION — Model: {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────
    df, fully_sampled_df, _ = load_and_separate_dataset()

    if len(fully_sampled_df) == 0:
        print("❌ No fully sampled files found! Exiting.")
        return

    # Get all unique subjects
    all_subjects = sorted(fully_sampled_df['subject'].unique())
    print(f"\n📊 Dataset Summary:")
    print(f"  Total fully sampled files: {len(fully_sampled_df)}")
    print(f"  Unique subjects: {len(all_subjects)}")
    print(f"  Subjects: {all_subjects}")

    # ── Cross-validation loop ──────────────────────────────
    cv_results = {}
    all_histories = {}

    for fold_idx, val_subject in enumerate(all_subjects):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(all_subjects)}")
        print(f"Validation subject: {val_subject}")
        print(f"{'='*60}")

        # Split data
        fold_train_df = fully_sampled_df[fully_sampled_df['subject'] != val_subject].reset_index(drop=True)
        fold_val_df = fully_sampled_df[fully_sampled_df['subject'] == val_subject].reset_index(drop=True)

        print(f"  Train: {len(fold_train_df)} files ({fold_train_df['subject'].nunique()} subjects)")
        print(f"  Val:   {len(fold_val_df)} files ({fold_val_df['subject'].nunique()} subjects)")
        print(f"  Train subjects: {sorted(fold_train_df['subject'].unique())}")

        # Create fresh model for each fold
        model = model_factories[model_name]()
        model_name_fold = f"{model_name}_fold{fold_idx+1}_val{val_subject}"

        # Train
        trained_model, history = run_training(
            train_df=fold_train_df,
            val_df=fold_val_df,
            model=model,
            model_name=model_name_fold,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device
        )

        # Store results
        best_ssim = max(history['val_ssim']) if history['val_ssim'] else 0
        best_psnr = max(history['val_psnr']) if history['val_psnr'] else 0
        best_epoch = history['val_ssim'].index(best_ssim) + 1 if history['val_ssim'] else 0

        cv_results[val_subject] = {
            'fold': fold_idx + 1,
            'val_subject': val_subject,
            'best_ssim': best_ssim,
            'best_psnr': best_psnr,
            'best_epoch': best_epoch,
            'train_subjects': sorted(fold_train_df['subject'].unique()),
            'train_files': len(fold_train_df),
            'val_files': len(fold_val_df),
        }
        all_histories[val_subject] = history

        # Clean up
        del trained_model
        torch.cuda.empty_cache()

    # ── Print summary ──────────────────────────────────────
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION RESULTS — {model_name}")
    print("="*60)
    print(f"{'Fold':>5} {'Val Subject':>12} {'Best SSIM':>10} {'Best PSNR':>10} {'Best Epoch':>11}")
    print("-"*55)

    all_ssims = []
    all_psnrs = []

    for subj, res in cv_results.items():
        print(f"{res['fold']:>5} {subj:>12} {res['best_ssim']:>10.4f} {res['best_psnr']:>10.2f} {res['best_epoch']:>11}")
        all_ssims.append(res['best_ssim'])
        all_psnrs.append(res['best_psnr'])

    print("-"*55)
    print(f"{'Mean':>18} {np.mean(all_ssims):>10.4f} {np.mean(all_psnrs):>10.2f}")
    print(f"{'Std':>18}  {np.std(all_ssims):>10.4f}  {np.std(all_psnrs):>10.2f}")
    print(f"{'Min':>18}  {np.min(all_ssims):>10.4f}  {np.min(all_psnrs):>10.2f}")
    print(f"{'Max':>18}  {np.max(all_ssims):>10.4f}  {np.max(all_psnrs):>10.2f}")
    print("="*60)

    # ── Save results ──────────────────────────────────────
    output_file = os.path.join(OUTPUT_DIR, f'cv_results_{model_name}.json')
    with open(output_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\n💾 Results saved: {output_file}")

    # ── Save individual fold histories ────────────────────
    for subj, history in all_histories.items():
        hist_file = os.path.join(OUTPUT_DIR, f'history_{model_name}_val{subj}.json')
        with open(hist_file, 'w') as f:
            json.dump(history, f, indent=2)

    # ── Plot results ──────────────────────────────────────
    plot_cv_summary(cv_results, model_name)
    plot_cv_curves(all_histories, model_name)

    return cv_results


def plot_cv_summary(cv_results, model_name):
    """Plot cross-validation summary bar charts."""
    subjects = list(cv_results.keys())
    ssims = [cv_results[s]['best_ssim'] for s in subjects]
    psnrs = [cv_results[s]['best_psnr'] for s in subjects]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'LOSO Cross-Validation — {model_name}\nOSI² ONE 47mT Low-Field MRI', fontsize=13, fontweight='bold')

    for ax, vals, ylabel, title in zip(
        axes,
        [ssims, psnrs],
        ['Best Val SSIM', 'Best Val PSNR (dB)'],
        ['SSIM per Subject', 'PSNR per Subject']
    ):
        colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
        bars = ax.bar(subjects, vals, color=colors, edgecolor='white', linewidth=1.5)

        mean_val = np.mean(vals)
        std_val = np.std(vals)

        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.4f}')
        ax.fill_between([-0.5, len(subjects)-0.5], mean_val - std_val, mean_val + std_val,
                        alpha=0.1, color='red', label=f'±1 std')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Held-out Subject (Validation)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'cv_summary_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📊 Plot saved: {save_path}")


def plot_cv_curves(all_histories, model_name):
    """Plot training curves across all folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Validation Curves Across Folds — {model_name}', fontsize=13, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    for ax, metric, ylabel in zip(
        axes,
        ['val_ssim', 'val_psnr'],
        ['SSIM', 'PSNR (dB)']
    ):
        for color, (subj, history) in zip(colors, all_histories.items()):
            if metric in history:
                ax.plot(history[metric], color=color, linewidth=1.5, label=f'Val={subj}', alpha=0.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} per Fold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'cv_curves_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📈 Plot saved: {save_path}")


if __name__ == "__main__":
    run_cross_validation(model_name='CascadeNet', num_epochs=2)