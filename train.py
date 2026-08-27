# train.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from config import CHECKPOINT_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from data_loader import MRIDataset
from utils import compute_psnr_ssim, kspace_to_image_magnitude


METRIC_DOMAIN = 'image_magnitude_3d_v1'


def combined_loss_fn(pred, target):
    """Image-prioritized reconstruction loss.

    Models in this project output k-space, but reconstruction quality is judged
    in image space. Keep a small k-space term for global consistency while making
    image-domain magnitude fidelity the dominant objective.
    """
    kspace_loss = F.smooth_l1_loss(pred, target)
    pred_img = kspace_to_image_magnitude(pred)
    target_img = kspace_to_image_magnitude(target)
    image_loss = F.l1_loss(pred_img, target_img)
    return 0.2 * kspace_loss + 0.8 * image_loss


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
                  device='cuda', resume=False, checkpoint_dir=None,
                  early_stopping_patience=20, early_stopping_min_delta=1e-4):
    """
    checkpoint_dir: where to save/load this fold's checkpoint + per-epoch history.
                    Defaults to config.CHECKPOINT_DIR if not given.
    """
    # Training folds use fresh random ky-kz masks per volume (augmentation);
    # validation folds use fixed per-file masks (deterministic metrics).
    train_set = MRIDataset(train_df, acceleration=acceleration, random_masks=True)
    val_set = MRIDataset(val_df, acceleration=acceleration, random_masks=False)
    # Volumes have different kz depths, so they cannot be stacked into a batch.
    # Keep the public batch_size argument for CLI compatibility, but process
    # one complete volume at a time.
    if batch_size != 1:
        print(f"  Using batch_size=1 for variable-depth 3D volumes (requested {batch_size})")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_ssim, best_epoch, start_epoch = 0.0, 0, 0
    epochs_without_improvement = 0
    history = {'metric_domain': METRIC_DOMAIN, 'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'grad_norms': [], 'lr': []}
    resume_checkpoint = resume

    ckpt_dir = str(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'best_{model_name}.pt')
    hist_path = os.path.join(ckpt_dir, f'history_{model_name}.json')

    # Restore prior history whenever it exists — even without a checkpoint
    # (a fold can be killed mid-epoch-1 before the first checkpoint save).
    if resume and os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                loaded_history = json.load(f)
            if loaded_history.get('metric_domain') == METRIC_DOMAIN:
                history = loaded_history
            else:
                resume_checkpoint = False
                print(f"⚠️  Ignoring incompatible history metrics in {hist_path}; "
                      f"expected {METRIC_DOMAIN}")
        except (json.JSONDecodeError, OSError):
            resume_checkpoint = False
            print(f"⚠️  Could not read {hist_path} — starting fresh history")

    if resume_checkpoint and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get('metric_domain') != METRIC_DOMAIN:
            print(f"⚠️  Ignoring incompatible checkpoint metrics in {ckpt_path}; "
                  f"expected {METRIC_DOMAIN}")
        else:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_epoch = start_epoch
            best_ssim = ckpt.get('val_ssim', 0.0)
            # Restore scheduler state so the LR trajectory matches an uninterrupted run.
            if 'scheduler_state' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            else:
                for _ in range(start_epoch):
                    scheduler.step()
            # The checkpoint is authoritative: trim any history entries written for
            # epochs beyond it (crash between history-write and checkpoint-save).
            for k, v in history.items():
                if isinstance(v, list):
                    history[k] = v[:start_epoch]
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

        if val_ssim > best_ssim + early_stopping_min_delta:
            best_ssim, best_epoch = val_ssim, epoch + 1
            epochs_without_improvement = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'metric_domain': METRIC_DOMAIN,
                        'val_ssim': val_ssim, 'val_psnr': val_psnr}, ckpt_path)
            print(f"  --> best model saved (SSIM={best_ssim:.4f})")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"  --> early stopping: no SSIM improvement ≥ {early_stopping_min_delta:g} "
                      f"for {early_stopping_patience} epochs; best epoch={best_epoch}")
                break

    return model, history