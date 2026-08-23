# train.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import CHECKPOINT_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from data_loader import MRIDataset
from utils import compute_psnr_ssim


def combined_loss_fn(pred, target):
    """70% k-space SmoothL1 + 30% image-domain L1."""
    kspace_loss = F.smooth_l1_loss(pred, target)
    def ks2mag(t):
        kc = t[:, 0] + 1j * t[:, 1]
        img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kc, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
        return img.abs()
    pred_img, target_img = ks2mag(pred), ks2mag(target)
    return 0.7 * kspace_loss + 0.3 * F.l1_loss(pred_img, target_img)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    grad_norms = []
    for inp, tgt, mask in tqdm(loader, desc="Train", leave=False):
        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(inp, mask)
        loss = loss_fn(out, tgt)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(float(gn))
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader), float(np.mean(grad_norms)) if grad_norms else 0.0


def validate(model, loader, device):
    model.eval()
    total_psnr, total_ssim, n_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for inp, tgt, mask in tqdm(loader, desc="Val", leave=False):
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            out = model(inp, mask)
            p, s = compute_psnr_ssim(out, tgt)
            bs = inp.shape[0]
            # Sample-weighted average so the last (smaller) batch isn't over-weighted.
            total_psnr += p * bs
            total_ssim += s * bs
            n_samples += bs
    return total_psnr / n_samples, total_ssim / n_samples


def run_training(train_df, val_df, model, model_name='model', acceleration=2,
                  num_epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                  device='cuda', resume=False, checkpoint_dir=None):
    """
    checkpoint_dir: where to save/load this fold's checkpoint + per-epoch history.
                    Defaults to config.CHECKPOINT_DIR if not given.
    """
    # Training folds use fresh random masks per slice (augmentation);
    # validation folds use fixed per-file masks (deterministic metrics).
    train_set = MRIDataset(train_df, acceleration=acceleration, random_masks=True)
    val_set = MRIDataset(val_df, acceleration=acceleration, random_masks=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=2, persistent_workers=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_ssim, best_epoch, start_epoch = 0.0, 0, 0
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'grad_norms': [], 'lr': []}

    ckpt_dir = str(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'best_{model_name}.pt')
    hist_path = os.path.join(ckpt_dir, f'history_{model_name}.json')

    # Restore prior history whenever it exists — even without a checkpoint
    # (a fold can be killed mid-epoch-1 before the first checkpoint save).
    if resume and os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"⚠️  Could not read {hist_path} — starting fresh history")

    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ssim = ckpt.get('val_ssim', 0.0)
        # Restore scheduler state so the LR trajectory matches an uninterrupted run.
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        # The checkpoint is authoritative: trim any history entries written for
        # epochs beyond it (crash between history-write and checkpoint-save).
        for k in history:
            history[k] = history[k][:start_epoch]
        print(f"Resumed from epoch {start_epoch}, best SSIM={best_ssim:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, avg_grad_norm = train_one_epoch(model, train_loader, optimizer, combined_loss_fn, device)
        val_psnr, val_ssim = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['grad_norms'].append(avg_grad_norm)
        history['lr'].append(lr)

        # Persist history after EVERY epoch — survives a timeout even if this
        # epoch never improves SSIM (and thus never triggers a checkpoint save below)
        with open(hist_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"  Loss: {train_loss:.6f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | GradNorm: {avg_grad_norm:.4f}")

        if val_ssim > best_ssim:
            best_ssim, best_epoch = val_ssim, epoch + 1
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'val_ssim': val_ssim, 'val_psnr': val_psnr}, ckpt_path)
            print(f"  --> best model saved (SSIM={best_ssim:.4f})")

    return model, history