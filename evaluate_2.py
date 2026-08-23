"""
evaluate_2.py

Consolidated post-CV evaluation suite, now run_manager-aware: points at a
SPECIFIC run's folder (checkpoints + results) rather than a flat global
CHECKPOINT_DIR/OUTPUT_DIR, since every main_cv.py execution now lives in its
own timestamped folder under RUNS_BASE_DIR.

Usage:
    python evaluate_2.py --task all --model CascadeNet                # latest run
    python evaluate_2.py --task all --model CascadeNet --run_id <id>  # specific run

Requires cv_results_{model}.json and best_{model}_fold{n}_val{subj}.pt to
already exist inside the target run's results/checkpoints folders (i.e.
main_cv.py has completed, fully or partially, for that run).
"""
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from config import DEFAULT_ACCELERATION
from data_loader import load_and_separate_dataset, ValMRIDataset
from model_registry import model_factories
from run_manager import load_run, latest_run

RUNS_BASE_DIR = os.path.expanduser("~/scratch/MRI_DATASET/Nelson_runs")

R_VALUES = [2, 4, 6, 8, 10, 12]


# ============================================================
# SHARED HELPERS
# ============================================================

def _mag(t):
    return torch.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2 + 1e-8)


def per_slice_metrics(pred, target):
    pred_mag, target_mag = _mag(pred), _mag(target)
    psnr_vals, ssim_vals = [], []
    for i in range(pred_mag.shape[0]):
        p, t = pred_mag[i].detach().cpu().numpy(), target_mag[i].detach().cpu().numpy()
        # Normalize by the TARGET's range only; apply the same scale to the
        # prediction so absolute intensity errors are preserved.
        t_min, rng_ = t.min(), t.max() - t.min() + 1e-8
        p, t = (p - t_min) / rng_, (t - t_min) / rng_
        psnr_vals.append(float(psnr(t, p, data_range=1)))
        ssim_vals.append(float(ssim(t, p, data_range=1)))
    return psnr_vals, ssim_vals


def kspace_to_image(kspace_2ch):
    kc = kspace_2ch[0] + 1j * kspace_2ch[1]
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kc)))
    return np.abs(img)


def norm_img(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def load_fold_checkpoint(run, model_name, fold_idx, val_subject, device):
    model_name_fold = f"{model_name}_fold{fold_idx}_val{val_subject}"
    ckpt_path = run.checkpoint_path(f'best_{model_name_fold}.pt')
    if not os.path.exists(ckpt_path):
        return None, None
    model = model_factories[model_name]()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    return model, ckpt


def get_cv_results(run, model_name):
    path = run.result_path(f'cv_results_{model_name}.json')
    with open(path) as f:
        return json.load(f)


# ============================================================
# TASK 1: RELIABILITY SWEEP (R=2..12) vs zero-filled
# ============================================================

def run_sweep(run, model_name, device):
    cv_results = get_cv_results(run, model_name)
    _, fully_sampled_df, _ = load_and_separate_dataset()

    results = {R: {'model_ssim': [], 'model_psnr': [], 'zf_ssim': [], 'zf_psnr': [], 'subjects': []}
               for R in R_VALUES}

    for val_subject, res in cv_results.items():
        fold_idx = res['fold']
        model, ckpt = load_fold_checkpoint(run, model_name, fold_idx, val_subject, device)
        if model is None:
            print(f"  ⚠️  Missing checkpoint for fold {fold_idx} (val={val_subject}) — skipping")
            continue

        fold_val_df = fully_sampled_df[fully_sampled_df['subject'] == val_subject].reset_index(drop=True)
        print(f"Fold {fold_idx} (val={val_subject}): sweeping R={R_VALUES}")

        for R in R_VALUES:
            val_set = ValMRIDataset(fold_val_df, acceleration=R)
            val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

            fold_model_ssim, fold_model_psnr, fold_zf_ssim, fold_zf_psnr = [], [], [], []
            with torch.no_grad():
                for inp, tgt, mask in val_loader:
                    inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                    out = model(inp, mask)
                    p, s = per_slice_metrics(out, tgt)
                    zp, zs = per_slice_metrics(inp, tgt)
                    fold_model_psnr.extend(p); fold_model_ssim.extend(s)
                    fold_zf_psnr.extend(zp); fold_zf_ssim.extend(zs)

            results[R]['model_ssim'].append(float(np.mean(fold_model_ssim)))
            results[R]['model_psnr'].append(float(np.mean(fold_model_psnr)))
            results[R]['zf_ssim'].append(float(np.mean(fold_zf_ssim)))
            results[R]['zf_psnr'].append(float(np.mean(fold_zf_psnr)))
            results[R]['subjects'].append(val_subject)

    out_path = run.result_path(f'reliability_sweep_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    for metric, zf_key, ylabel in [('model_ssim', 'zf_ssim', 'SSIM'), ('model_psnr', 'zf_psnr', 'PSNR (dB)')]:
        fig, ax = plt.subplots(figsize=(9, 6))
        means = [np.mean(results[R][metric]) for R in R_VALUES]
        stds = [np.std(results[R][metric]) for R in R_VALUES]
        zf_means = [np.mean(results[R][zf_key]) for R in R_VALUES]

        ax.plot(R_VALUES, means, '-o', color='#4C72B0', label=model_name)
        ax.fill_between(R_VALUES, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                         alpha=0.15, color='#4C72B0')
        ax.plot(R_VALUES, zf_means, '--', color='gray', label='Zero-filled baseline')
        ax.axvline(8, color='red', linestyle=':', alpha=0.5, label='R=8 (reliability boundary)')
        ax.set_xlabel('Acceleration Factor (R)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs. Acceleration Factor — {model_name} (fold-fair LOSO)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        save_path = os.path.join(run.figures, f'reliability_sweep_{model_name}_{ylabel.split()[0]}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


# ============================================================
# TASK 2: PER-SUBJECT BOXPLOT (slice-level distribution)
# ============================================================

def run_boxplot(run, model_name, device, acceleration=DEFAULT_ACCELERATION):
    cv_results = get_cv_results(run, model_name)
    _, fully_sampled_df, _ = load_and_separate_dataset()

    subject_ssim, subject_psnr, zf_subject_ssim, zf_subject_psnr = {}, {}, {}, {}

    for val_subject, res in cv_results.items():
        fold_idx = res['fold']
        model, ckpt = load_fold_checkpoint(run, model_name, fold_idx, val_subject, device)
        if model is None:
            continue

        fold_val_df = fully_sampled_df[fully_sampled_df['subject'] == val_subject].reset_index(drop=True)
        val_set = ValMRIDataset(fold_val_df, acceleration=acceleration)
        val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

        ssims, psnrs, zf_ssims, zf_psnrs = [], [], [], []
        with torch.no_grad():
            for inp, tgt, mask in val_loader:
                inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
                out = model(inp, mask)
                p, s = per_slice_metrics(out, tgt)
                zp, zs = per_slice_metrics(inp, tgt)
                psnrs.extend(p); ssims.extend(s)
                zf_psnrs.extend(zp); zf_ssims.extend(zs)

        subject_ssim[val_subject] = ssims
        subject_psnr[val_subject] = psnrs
        zf_subject_ssim[val_subject] = zf_ssims
        zf_subject_psnr[val_subject] = zf_psnrs
        print(f"  Subj {val_subject}: model SSIM={np.mean(ssims):.4f}, ZF SSIM={np.mean(zf_ssims):.4f}")

    out = {'model_ssim': subject_ssim, 'model_psnr': subject_psnr,
           'zf_ssim': zf_subject_ssim, 'zf_psnr': zf_subject_psnr}
    out_path = run.result_path(f'per_subject_distributions_{model_name}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")

    for metric_key, ylabel in [('ssim', 'SSIM'), ('psnr', 'PSNR (dB)')]:
        model_data = out[f'model_{metric_key}']
        zf_data = out[f'zf_{metric_key}']
        subjects = list(model_data.keys())

        fig, ax = plt.subplots(figsize=(max(10, len(subjects) * 1.5), 6))
        pos_model = np.arange(len(subjects)) * 2.0
        pos_zf = pos_model + 0.7

        bp1 = ax.boxplot([model_data[s] for s in subjects], positions=pos_model, widths=0.6,
                          patch_artist=True, showfliers=False)
        bp2 = ax.boxplot([zf_data[s] for s in subjects], positions=pos_zf, widths=0.6,
                          patch_artist=True, showfliers=False)
        for box in bp1['boxes']: box.set_facecolor('#4C72B0'); box.set_alpha(0.7)
        for box in bp2['boxes']: box.set_facecolor('#C4C4C4'); box.set_alpha(0.7)

        ax.set_xticks(pos_model + 0.35)
        ax.set_xticklabels(subjects)
        ax.set_xlabel('Held-out Subject')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{model_name} — Per-slice {ylabel} by Held-out Subject (model vs. zero-filled)')
        ax.legend([bp1['boxes'][0], bp2['boxes'][0]], [model_name, 'Zero-filled'], loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

        save_path = os.path.join(run.figures, f'per_subject_boxplot_{model_name}_{metric_key}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


# ============================================================
# TASK 3: PAIRED STATISTICAL SIGNIFICANCE
# ============================================================

def _paired_test(va, vb, label_a, label_b):
    va, vb = np.array(va), np.array(vb)
    diffs = va - vb
    w_stat, w_p = scipy_stats.wilcoxon(va, vb)
    t_stat, t_p = scipy_stats.ttest_rel(va, vb)
    return {
        'label_a': label_a, 'label_b': label_b, 'n_folds': len(va),
        'mean_a': float(np.mean(va)), 'mean_b': float(np.mean(vb)),
        'mean_diff': float(np.mean(diffs)), 'std_diff': float(np.std(diffs)),
        'wilcoxon_stat': float(w_stat), 'wilcoxon_p': float(w_p),
        'ttest_stat': float(t_stat), 'ttest_p': float(t_p),
        'significant_wilcoxon': bool(w_p < 0.05), 'significant_ttest': bool(t_p < 0.05),
    }


def _print_result(r):
    print(f"\n{r['label_a']} vs {r['label_b']}  (n={r['n_folds']} paired folds)")
    print(f"  Mean {r['label_a']}: {r['mean_a']:.4f}   Mean {r['label_b']}: {r['mean_b']:.4f}   "
          f"diff: {r['mean_diff']:+.4f}")
    print(f"  Wilcoxon p={r['wilcoxon_p']:.4f} {'✅' if r['significant_wilcoxon'] else '❌'}   "
          f"Paired t-test p={r['ttest_p']:.4f} {'✅' if r['significant_ttest'] else '❌'}  (α=0.05)")


def run_stats(run, model_names, device):
    """NOTE: for cross-architecture comparison, all model_names must belong to
    the SAME run (i.e. you trained multiple architectures within one run_id).
    If you train architectures in separate runs, pass --run_id explicitly and
    re-run per architecture, then compare cv_results JSONs manually."""
    all_results = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            a, b = model_names[i], model_names[j]
            ra, rb = get_cv_results(run, a), get_cv_results(run, b)
            common = sorted(set(ra) & set(rb))
            r = _paired_test([ra[s]['best_ssim'] for s in common],
                              [rb[s]['best_ssim'] for s in common], a, b)
            r['subjects_compared'] = common
            _print_result(r)
            all_results[f"{a}_vs_{b}"] = r

    for model_name in model_names:
        sweep_path = run.result_path(f'reliability_sweep_{model_name}.json')
        if not os.path.exists(sweep_path):
            print(f"⚠️  No reliability_sweep_{model_name}.json — run 'sweep' task first for zero-filled comparison")
            continue
        with open(sweep_path) as f:
            sweep = json.load(f)
        R_entry = sweep[str(DEFAULT_ACCELERATION)]
        zf_by_subject = dict(zip(R_entry['subjects'], R_entry['zf_ssim']))
        # Use the sweep's model SSIM (same eval protocol/masks as the ZF numbers)
        # rather than cv_results' best-epoch SSIM, which is selection-biased.
        sweep_model_by_subject = dict(zip(R_entry['subjects'], R_entry['model_ssim']))
        cv = get_cv_results(run, model_name)
        common = sorted(set(sweep_model_by_subject) & set(zf_by_subject))
        r = _paired_test([sweep_model_by_subject[s] for s in common],
                          [zf_by_subject[s] for s in common], model_name, 'Zero-filled')
        r['subjects_compared'] = common
        _print_result(r)
        all_results[f"{model_name}_vs_zerofilled"] = r

    out_path = run.result_path('significance_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Saved: {out_path}")


# ============================================================
# TASK 4: FAIR RECONSTRUCTION VISUALIZATIONS (top folds)
# ============================================================

def run_visualize(run, model_name, device, acceleration=DEFAULT_ACCELERATION, top_n=4):
    cv_results = get_cv_results(run, model_name)
    ranked = sorted(cv_results.items(), key=lambda kv: kv[1]['best_ssim'], reverse=True)
    top_folds = ranked[:top_n]
    print(f"Top {top_n} folds by SSIM: {[(s, r['best_ssim']) for s, r in top_folds]}")

    _, fully_sampled_df, _ = load_and_separate_dataset()
    fig, axes = plt.subplots(top_n, 4, figsize=(16, 4 * top_n))
    if top_n == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (val_subject, res) in enumerate(top_folds):
        fold_idx = res['fold']
        model, ckpt = load_fold_checkpoint(run, model_name, fold_idx, val_subject, device)
        if model is None:
            continue

        fold_val_df = fully_sampled_df[fully_sampled_df['subject'] == val_subject].reset_index(drop=True)
        val_set = ValMRIDataset(fold_val_df, acceleration=acceleration)

        n_slices = val_set.slices_per_file[0]
        global_idx = n_slices // 2
        inp, tgt, mask = val_set[global_idx]

        with torch.no_grad():
            inp_b, tgt_b, mask_b = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
            out_b = model(inp_b, mask_b)
            p_list, s_list = per_slice_metrics(out_b, tgt_b)
            zp_list, zs_list = per_slice_metrics(inp_b, tgt_b)

        inp_img = norm_img(kspace_to_image(inp.numpy()))
        out_img = norm_img(kspace_to_image(out_b[0].cpu().numpy()))
        tgt_img = norm_img(kspace_to_image(tgt.numpy()))
        err_img = np.abs(out_img - tgt_img)

        axes[row_idx, 0].imshow(inp_img, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f'Zero-filled\nSubj {val_subject} (fold {fold_idx})\n'
                                    f'PSNR={zp_list[0]:.2f}dB SSIM={zs_list[0]:.4f}', fontsize=9)
        axes[row_idx, 0].axis('off')

        axes[row_idx, 1].imshow(out_img, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f'{model_name} output\n(never trained on Subj {val_subject})\n'
                                    f'PSNR={p_list[0]:.2f}dB SSIM={s_list[0]:.4f}', fontsize=9)
        axes[row_idx, 1].axis('off')

        axes[row_idx, 2].imshow(tgt_img, cmap='gray', vmin=0, vmax=1)
        axes[row_idx, 2].set_title('Ground truth', fontsize=9)
        axes[row_idx, 2].axis('off')

        im = axes[row_idx, 3].imshow(err_img, cmap='hot', vmin=0, vmax=err_img.max())
        axes[row_idx, 3].set_title('|Output − GT|', fontsize=9)
        axes[row_idx, 3].axis('off')
        plt.colorbar(im, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

    fig.suptitle(f'{model_name} — Fair LOSO Reconstruction Examples (R={acceleration})\n'
                 f'Each row: fold-specific checkpoint evaluated only on its own held-out subject',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(run.figures, f'fold_reconstructions_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-CV evaluation suite (run_manager-aware)")
    parser.add_argument('--task', choices=['sweep', 'boxplot', 'stats', 'visualize', 'all'], required=True)
    parser.add_argument('--model', type=str, default='CascadeNet', help='Model name for sweep/boxplot/visualize')
    parser.add_argument('--models', type=str, nargs='+', default=['CascadeNet'],
                         help='Model names for stats (2+ for architecture comparison, must be in the SAME run)')
    parser.add_argument('--acceleration', type=int, default=DEFAULT_ACCELERATION)
    parser.add_argument('--top_n', type=int, default=4)
    parser.add_argument('--run_id', type=str, default=None,
                         help='Which run to evaluate. Defaults to the most recent run for --model.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.run_id:
        run = load_run(args.run_id, base_dir=RUNS_BASE_DIR)
    else:
        rid = latest_run(RUNS_BASE_DIR, model_name=args.model)
        run = load_run(rid, base_dir=RUNS_BASE_DIR)
    print(f"Evaluating run: {run.run_id}")

    if args.task in ('sweep', 'all'):
        run_sweep(run, args.model, device)
    if args.task in ('boxplot', 'all'):
        run_boxplot(run, args.model, device, acceleration=args.acceleration)
    if args.task in ('stats', 'all'):
        run_stats(run, args.models, device)
    if args.task in ('visualize', 'all'):
        run_visualize(run, args.model, device, acceleration=args.acceleration, top_n=args.top_n)