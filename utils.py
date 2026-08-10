# utils.py
import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from config import N_PE, N_RO

def load_kspace_optimized(filepath, verbose=False):
    """Load k-space and pad PE dimension to 136."""
    with h5py.File(filepath, 'r') as f:
        data = f['dataset']['data'][()]
    if len(data) == 0:
        raise ValueError(f"Empty data in {filepath}")

    pe_indices, slice_indices, n_samples_list = [], [], []
    for line in data:
        head = line['head']
        pe_indices.append(int(head['idx']['kspace_encode_step_1']))
        slice_indices.append(int(head['idx']['kspace_encode_step_2']))
        n_samples_list.append(int(head['number_of_samples']))

    unique_pe = sorted(set(pe_indices))
    unique_slices = sorted(set(slice_indices))
    n_pe, n_slices, n_samples = len(unique_pe), len(unique_slices), n_samples_list[0]

    pe_to_idx = {pe: i for i, pe in enumerate(unique_pe)}
    slice_to_idx = {sl: i for i, sl in enumerate(unique_slices)}

    kspace = np.zeros((n_slices, n_pe, n_samples), dtype=np.complex64)
    for line in data:
        head = line['head']
        pe_idx = pe_to_idx[int(head['idx']['kspace_encode_step_1'])]
        sl_idx = slice_to_idx[int(head['idx']['kspace_encode_step_2'])]
        raw = line['data']
        raw_complex = raw[0::2] + 1j * raw[1::2]
        kspace[sl_idx, pe_idx, :] = raw_complex[:n_samples]

    kspace = pad_to_136(kspace)
    return kspace

def pad_to_136(kspace_3d):
    """
    Pad k-space to 136 PE lines.
    Assumes input shape: (slices, PE, readout)
    Returns: (slices, 136, readout)
    """
    current_pe = kspace_3d.shape[1]
    
    if current_pe == 136:
        return kspace_3d
    elif current_pe < 136:
        pad_amount = 136 - current_pe
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        return np.pad(kspace_3d, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')
    else:
        crop = (current_pe - 136) // 2
        return kspace_3d[:, crop:crop+136, :]
        
def normalize_kspace(kspace):
    """Normalize k-space by max magnitude."""
    scale = np.abs(kspace).max() + 1e-9
    return (kspace / scale).astype(np.complex64), scale

def compute_psnr_ssim(pred, target):
    """Compute PSNR and SSIM from 2-channel tensors."""
    pred_mag = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2 + 1e-8)
    target_mag = torch.sqrt(target[:, 0]**2 + target[:, 1]**2 + 1e-8)
    psnr_vals, ssim_vals = [], []
    for i in range(pred_mag.shape[0]):
        p, t = pred_mag[i].cpu().numpy(), target_mag[i].cpu().numpy()
        p, t = (p - p.min()) / (p.max() - p.min() + 1e-8), (t - t.min()) / (t.max() - t.min() + 1e-8)
        psnr_vals.append(psnr(t, p, data_range=1))
        ssim_vals.append(ssim(t, p, data_range=1))
    return np.mean(psnr_vals), np.mean(ssim_vals)