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
from data_loader import TrainMRIDataset, ValMRIDataset
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
    for inp, tgt, mask in tqdm(loader, desc="Train", leave=False):
        inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(inp, mask)
        loss = loss_fn(out, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_psnr, total_ssim = 0, 0
    with torch.no_grad():
        for inp, tgt, mask in tqdm(loader, desc="Val", leave=False):
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            out = model(inp, mask)
            p, s = compute_psnr_ssim(out, tgt)
            total_psnr += p
            total_ssim += s
    n = len(loader)
    return total_psnr / n, total_ssim / n

def run_training(train_df, val_df, model, model_name='model', acceleration=2, num_epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, device='cuda', resume=False):
    train_set = TrainMRIDataset(train_df, acceleration=acceleration)
    val_set = ValMRIDataset(val_df, acceleration=acceleration)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_ssim, best_epoch, start_epoch = 0.0, 0, 0
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': [], 'grad_norms': [], 'lr': []}

    ckpt_path = os.path.join(CHECKPOINT_DIR, f'best_{model_name}.pt')
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ssim = ckpt.get('val_ssim', 0.0)
        print(f"Resumed from epoch {start_epoch}, best SSIM={best_ssim:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, combined_loss_fn, device)
        val_psnr, val_ssim = validate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)
        history['lr'].append(lr)

        print(f"  Loss: {train_loss:.6f} | PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f}")

        if val_ssim > best_ssim:
            best_ssim, best_epoch = val_ssim, epoch + 1
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                        'val_ssim': val_ssim, 'val_psnr': val_psnr}, ckpt_path)
            print(f"  --> best model saved (SSIM={best_ssim:.4f})")

    return model, history