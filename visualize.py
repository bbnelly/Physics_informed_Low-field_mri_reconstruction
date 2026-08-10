# visualize.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from tqdm import tqdm
from config import OUTPUT_DIR

def kspace_to_image(kspace_2ch):
    """Convert 2-channel k-space to magnitude image."""
    kc = kspace_2ch[0] + 1j * kspace_2ch[1]
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kc)))
    return np.abs(img)


def norm_img(img):
    """Normalize image to [0, 1]."""
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def generate_reconstruction_pdf(trained_models_dict, val_loader, device, all_results,
                                pdf_path=None, n_samples=8):
    """
    Generate PDF showing all models side by side on same samples.
    Columns: Zero-filled input | Model 1 | Model 2 | ... | Ground Truth | Error Map
    """
    if pdf_path is None:
        pdf_path = os.path.join(OUTPUT_DIR, 'reconstruction_comparison.pdf')

    model_names = list(trained_models_dict.keys())
    n_models = len(model_names)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']

    for m in trained_models_dict.values():
        m.eval()

    def metrics(pred, tgt):
        p = norm_img(pred)
        t = norm_img(tgt)
        psnr_val = float(np.clip(psnr(t, p, data_range=1.0), 0, 60))
        ssim_val = float(np.clip(ssim(t, p, data_range=1.0), 0, 1))
        return psnr_val, ssim_val

    # Collect samples
    samples = []
    with torch.no_grad():
        for inp, tgt, mask in val_loader:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            outputs = {name: model(inp, mask) for name, model in trained_models_dict.items()}

            for i in range(inp.shape[0]):
                if len(samples) >= n_samples:
                    break

                inp_img = kspace_to_image(inp[i].cpu().numpy())
                tgt_img = kspace_to_image(tgt[i].cpu().numpy())
                psnr_zf, ssim_zf = metrics(inp_img, tgt_img)

                model_outputs = {}
                for name in model_names:
                    out_img = kspace_to_image(outputs[name][i].cpu().numpy())
                    p, s = metrics(out_img, tgt_img)
                    model_outputs[name] = {'img': out_img, 'psnr': p, 'ssim': s}

                best_name = max(model_outputs, key=lambda n: model_outputs[n]['ssim'])
                err_img = np.abs(norm_img(model_outputs[best_name]['img']) - norm_img(tgt_img))

                samples.append({
                    'inp': inp_img, 'tgt': tgt_img, 'err': err_img,
                    'best_name': best_name, 'psnr_zf': psnr_zf, 'ssim_zf': ssim_zf,
                    'model_outputs': model_outputs
                })

            if len(samples) >= n_samples:
                break

    print(f"Collected {len(samples)} samples")

    # Build PDF
    with PdfPages(pdf_path) as pdf:
        # Cover page
        fig = plt.figure(figsize=(14, 10))
        plt.axis('off')
        fig.text(0.5, 0.92, 'Multi-Model Reconstruction Comparison\nOSI² ONE 47mT Low-Field MRI',
                 ha='center', fontsize=16, fontweight='bold')

        table_data = [['Model', 'PSNR (dB)', 'SSIM', 'ΔPSNR vs ZF', 'ΔSSIM vs ZF', 'Params']]
        zf_psnr_mean = np.mean([s['psnr_zf'] for s in samples])
        zf_ssim_mean = np.mean([s['ssim_zf'] for s in samples])
        table_data.append(['Zero-fill', f"{zf_psnr_mean:.2f}", f"{zf_ssim_mean:.4f}", '—', '—', '—'])

        for name in model_names:
            psnrs = [s['model_outputs'][name]['psnr'] for s in samples]
            ssims = [s['model_outputs'][name]['ssim'] for s in samples]
            mean_psnr, mean_ssim = np.mean(psnrs), np.mean(ssims)
            table_data.append([
                name,
                f"{mean_psnr:.2f} ± {np.std(psnrs):.2f}",
                f"{mean_ssim:.4f} ± {np.std(ssims):.4f}",
                f"{mean_psnr - zf_psnr_mean:+.2f}",
                f"{mean_ssim - zf_ssim_mean:+.4f}",
                f"{all_results[name]['n_params']:,}"
            ])

        ax_table = fig.add_axes([0.05, 0.35, 0.90, 0.50])
        ax_table.axis('off')
        tbl = ax_table.table(cellText=table_data, cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 2.0)

        # Highlight best model
        best_overall = max(model_names, key=lambda n: np.mean([s['model_outputs'][n]['ssim'] for s in samples]))
        best_row = model_names.index(best_overall) + 2
        for col in range(len(table_data[0])):
            tbl[best_row, col].set_facecolor('#d4edda')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Per-sample pages
        n_cols = n_models + 3
        for sample_idx, s in enumerate(samples):
            fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 5))
            fig.suptitle(f'Sample {sample_idx + 1} / {len(samples)} | Best: {s["best_name"]}')

            # Zero-filled input
            ax = axes[0]
            ax.imshow(norm_img(s['inp']), cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Zero-filled\nPSNR={s["psnr_zf"]:.2f}dB\nSSIM={s["ssim_zf"]:.4f}', fontsize=8)
            ax.axis('off')

            # Each model
            for col_idx, name in enumerate(model_names):
                mo = s['model_outputs'][name]
                ax = axes[col_idx + 1]
                is_best = (name == s['best_name'])
                ax.imshow(norm_img(mo['img']), cmap='gray', vmin=0, vmax=1)
                ax.set_title(f"{'★ ' if is_best else ''}{name}\nPSNR={mo['psnr']:.2f}dB\nSSIM={mo['ssim']:.4f}",
                             fontsize=8, color='green' if is_best else 'black')
                ax.axis('off')
                if is_best:
                    for spine in ax.spines.values():
                        spine.set_edgecolor('gold')
                        spine.set_linewidth(3)
                        spine.set_visible(True)

            # Ground truth
            ax = axes[n_models + 1]
            ax.imshow(norm_img(s['tgt']), cmap='gray', vmin=0, vmax=1)
            ax.set_title('Ground Truth', fontsize=8, fontweight='bold')
            ax.axis('off')

            # Error map
            ax = axes[n_models + 2]
            im = ax.imshow(s['err'], cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Error\n({s["best_name"]} vs GT)', fontsize=8)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"PDF saved: {pdf_path}")
    return pdf_path


def plot_training_curves(history, save_path=None):
    """Plot training curves: loss, PSNR, SSIM, gradient norm."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'training_curves.png')

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves — OSI² ONE 47mT', fontsize=14, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', markersize=3)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # SSIM
    axes[0, 1].plot(epochs, history['val_ssim'], 'g-o', markersize=3)
    axes[0, 1].set_title('Validation SSIM')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].grid(True, alpha=0.3)

    # PSNR
    axes[1, 0].plot(epochs, history['val_psnr'], 'r-o', markersize=3)
    axes[1, 0].set_title('Validation PSNR')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].grid(True, alpha=0.3)

    # Gradient norms
    axes[1, 1].plot(epochs, history.get('grad_norms', [0]*len(epochs)), 'm-o', markersize=3)
    axes[1, 1].set_title('Gradient Norm')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('L2 Norm')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_comparison_table(all_results, save_path=None):
    """Generate comparison bar charts for all models."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')

    names = list(all_results.keys())
    ssims = [all_results[n]['best_ssim'] for n in names]
    psnrs = [all_results[n]['best_psnr'] for n in names]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison — OSI² ONE 47mT', fontsize=13, fontweight='bold')

    for ax, vals, ylabel, title in zip(axes, [ssims, psnrs], ['Best Val SSIM', 'Best Val PSNR (dB)'],
                                       ['SSIM Comparison', 'PSNR Comparison']):
        bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor='white', linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path