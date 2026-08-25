# main_cv.py
import os
import json
import logging
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import (DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_ACCELERATION,
                    DEFAULT_LEARNING_RATE, CV_SUBJECTS)
from data_loader import load_and_separate_dataset
from model_registry import model_factories
from train import METRIC_DOMAIN, run_training
from run_manager import setup_run, load_run, latest_run

RUNS_BASE_DIR = os.path.expanduser("~/scratch/MRI_DATASET/Nelson_runs")


def run_cross_validation(run, model_name='CascadeNet', num_epochs=DEFAULT_EPOCHS,
                        batch_size=DEFAULT_BATCH_SIZE, acceleration=DEFAULT_ACCELERATION):
    """
    Run Leave-One-Subject-Out Cross-Validation.
    Each subject takes turns being the validation set.

    ALL outputs (checkpoints, per-fold history, cv_results, figures) go into
    `run` (a RunPaths from run_manager.py) — either a fresh timestamped folder
    or a resumed prior run, decided by the caller in __main__.

    Pure 9-subject LOSO: every subject takes a turn as the held-out validation
    fold, and there is no separate permanent test set. (config.py still has
    TEST_SUBJECTS/TEST_SUBJECTS_CV variables left over from an earlier design —
    they are intentionally unused here.)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = logging.getLogger(run.run_id)

    print(f"\n{'='*60}")
    print(f"LOSO CROSS-VALIDATION — Model: {model_name}")
    print(f"Device: {device}")
    print(f"Run: {run.run_id}")
    print(f"{'='*60}")
    logger.info(f"LOSO CV starting — model={model_name}, epochs={num_epochs}, batch_size={batch_size}, lr={DEFAULT_LEARNING_RATE}")

    # ── Load data ────────────────────────────────────────────
    df, fully_sampled_df, _ = load_and_separate_dataset()

    if len(fully_sampled_df) == 0:
        print("❌ No fully sampled files found! Exiting.")
        return

    available_subjects = set(fully_sampled_df['subject'].unique())
    all_subjects = sorted(s for s in CV_SUBJECTS if s in available_subjects)

    print(f"\n📊 Dataset Summary:")
    print(f"  Total fully sampled files: {len(fully_sampled_df)}")
    print(f"  CV subjects (folds): {len(all_subjects)} — {all_subjects}")
    logger.info(f"CV subjects: {all_subjects}")

    # ── Cross-validation loop ──────────────────────────────
    cv_results = {}
    all_histories = {}

    for fold_idx, val_subject in enumerate(all_subjects):
        model_name_fold = f"{model_name}_fold{fold_idx+1}_val{val_subject}"
        ckpt_path = run.checkpoint_path(f'best_{model_name_fold}.pt')
        hist_path = run.checkpoint_path(f'history_{model_name_fold}.json')

        fold_train_df = fully_sampled_df[fully_sampled_df['subject'] != val_subject].reset_index(drop=True)
        fold_val_df = fully_sampled_df[fully_sampled_df['subject'] == val_subject].reset_index(drop=True)

        # Fully-done check: history has num_epochs entries (robust even if the
        # final epoch never beat best SSIM and so never re-saved the checkpoint)
        already_done = False
        existing_history = None
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                existing_history = json.load(f)
            if (existing_history.get('metric_domain') == METRIC_DOMAIN and
                    len(existing_history.get('train_loss', [])) >= num_epochs):
                already_done = True

        if already_done:
            print(f"⏭️  Fold {fold_idx+1} (val={val_subject}) already completed — skipping")
            logger.info(f"Fold {fold_idx+1} (val={val_subject}) already completed — skipping")
            history = existing_history or {'metric_domain': METRIC_DOMAIN, 'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'grad_norms': [], 'lr': []}
            if history['val_ssim']:
                best_idx = int(np.argmax(history['val_ssim']))
                best_ssim = history['val_ssim'][best_idx]
                best_psnr = history['val_psnr'][best_idx]
                best_epoch = best_idx + 1
            else:
                best_ssim, best_psnr, best_epoch = 0, 0, 0
            cv_results[val_subject] = {
                'fold': fold_idx + 1, 'val_subject': val_subject,
                'best_ssim': best_ssim, 'best_psnr': best_psnr, 'best_epoch': best_epoch,
                'train_subjects': sorted(fold_train_df['subject'].unique()),
                'train_files': len(fold_train_df), 'val_files': len(fold_val_df),
            }
            all_histories[val_subject] = history
            continue

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{len(all_subjects)}")
        print(f"Validation subject: {val_subject}")
        print(f"{'='*60}")
        logger.info(f"Fold {fold_idx+1}/{len(all_subjects)} — val subject {val_subject} — starting")

        print(f"  Train: {len(fold_train_df)} files ({fold_train_df['subject'].nunique()} subjects)")
        print(f"  Val:   {len(fold_val_df)} files ({fold_val_df['subject'].nunique()} subjects)")
        print(f"  Train subjects: {sorted(fold_train_df['subject'].unique())}")

        model = model_factories[model_name]()

        trained_model, history = run_training(
            train_df=fold_train_df,
            val_df=fold_val_df,
            model=model,
            model_name=model_name_fold,
            acceleration=acceleration,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            resume=True,                       # picks up mid-fold if a partial checkpoint exists
            checkpoint_dir=str(run.checkpoints),
        )

        if history['val_ssim']:
            best_idx = int(np.argmax(history['val_ssim']))
            best_ssim = history['val_ssim'][best_idx]
            best_psnr = history['val_psnr'][best_idx]
            best_epoch = best_idx + 1
        else:
            best_ssim, best_psnr, best_epoch = 0, 0, 0

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
        logger.info(f"Fold {fold_idx+1} done — best_ssim={best_ssim:.4f}, best_psnr={best_psnr:.2f}, best_epoch={best_epoch}")

        del trained_model
        torch.cuda.empty_cache()

    # ── Print summary ──────────────────────────────────────
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION RESULTS — {model_name}")
    print("="*60)
    print(f"{'Fold':>5} {'Val Subject':>12} {'Best SSIM':>10} {'Best PSNR':>10} {'Best Epoch':>11}")
    print("-"*55)

    all_ssims, all_psnrs = [], []
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

    # ── Save results (into run.results, not the old flat OUTPUT_DIR) ──
    output_file = run.result_path(f'cv_results_{model_name}.json')
    with open(output_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\n💾 Results saved: {output_file}")

    for subj, history in all_histories.items():
        hist_file = run.result_path(f'history_{model_name}_val{subj}.json')
        with open(hist_file, 'w') as f:
            json.dump(history, f, indent=2)

    # ── Plot results (into run.figures) ────────────────────
    plot_cv_summary(cv_results, model_name, run.figures)
    plot_cv_curves(all_histories, model_name, run.figures)

    logger.info("LOSO CV complete.")
    return cv_results


def plot_cv_summary(cv_results, model_name, figures_dir):
    subjects = list(cv_results.keys())
    ssims = [cv_results[s]['best_ssim'] for s in subjects]
    psnrs = [cv_results[s]['best_psnr'] for s in subjects]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'LOSO Cross-Validation — {model_name}\nOSI² ONE 47mT Low-Field MRI', fontsize=13, fontweight='bold')

    for ax, vals, ylabel, title in zip(
        axes, [ssims, psnrs], ['Best Val SSIM', 'Best Val PSNR (dB)'], ['SSIM per Subject', 'PSNR per Subject']
    ):
        colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
        bars = ax.bar(subjects, vals, color=colors, edgecolor='white', linewidth=1.5)
        mean_val, std_val = np.mean(vals), np.std(vals)
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.4f}')
        ax.fill_between([-0.5, len(subjects)-0.5], mean_val - std_val, mean_val + std_val,
                        alpha=0.1, color='red', label='±1 std')
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
    save_path = os.path.join(figures_dir, f'cv_summary_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot saved: {save_path}")


def plot_cv_curves(all_histories, model_name, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Validation Curves Across Folds — {model_name}', fontsize=13, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))

    for ax, metric, ylabel in zip(axes, ['val_ssim', 'val_psnr'], ['SSIM', 'PSNR (dB)']):
        for color, (subj, history) in zip(colors, all_histories.items()):
            if metric in history:
                ax.plot(history[metric], color=color, linewidth=1.5, label=f'Val={subj}', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} per Fold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, f'cv_curves_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 Plot saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO cross-validation")
    parser.add_argument('--model', type=str, default='CascadeNet')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--acceleration', type=int, default=DEFAULT_ACCELERATION,
                        help='Undersampling factor used for training and validation in this run')
    parser.add_argument('--resume', action='store_true',
                         help='Resume the most recent run for this model instead of starting fresh')
    parser.add_argument('--run_id', type=str, default=None,
                         help='Resume a SPECIFIC run_id instead of the latest')
    args = parser.parse_args()

    if args.run_id:
        run = load_run(args.run_id, base_dir=RUNS_BASE_DIR)
        print(f"Loaded existing run: {run.run_id}")
    elif args.resume:
        try:
            rid = latest_run(RUNS_BASE_DIR, model_name=args.model)
            run = load_run(rid, base_dir=RUNS_BASE_DIR)
            print(f"Resuming latest run: {run.run_id}")
        except FileNotFoundError:
            print("No prior run found for this model — starting fresh")
            run = setup_run(args.model, base_dir=RUNS_BASE_DIR,
                             extra_config={'num_epochs': args.epochs, 'batch_size': args.batch_size,
                                           'learning_rate': DEFAULT_LEARNING_RATE})
    else:
        run = setup_run(args.model, base_dir=RUNS_BASE_DIR,
                         extra_config={'num_epochs': args.epochs, 'batch_size': args.batch_size,
                                       'learning_rate': DEFAULT_LEARNING_RATE})

    run_cross_validation(run, model_name=args.model, num_epochs=args.epochs,
                         batch_size=args.batch_size, acceleration=args.acceleration)