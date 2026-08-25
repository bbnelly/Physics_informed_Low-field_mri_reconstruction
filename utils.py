# utils.py
import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


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

def kspace_to_image_magnitude(kspace_2ch):
    """Convert 2-channel k-space tensors to image-domain magnitude.

    Preferred Option-A shape is (B, 2, kz, ky, kx), reconstructed with 3D IFFT.
    A legacy (B, 2, ky, kx) 2D fallback is kept for older visualization helpers.
    """
    kc = kspace_2ch[:, 0] + 1j * kspace_2ch[:, 1]
    if kc.ndim == 4:
        img = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kc, dim=(-3, -2, -1)),
                            dim=(-3, -2, -1), norm='ortho'),
            dim=(-3, -2, -1),
        )
    elif kc.ndim == 3:
        img = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(kc, dim=(-2, -1)), norm='ortho'),
            dim=(-2, -1),
        )
    else:
        raise ValueError(f"Expected 4D/5D 2-channel k-space tensor, got shape {tuple(kspace_2ch.shape)}")
    return img.abs()


def compute_psnr_ssim(pred, target):
    """Compute image-domain PSNR/SSIM from 2-channel k-space tensors.

    For 3D volumes, metrics are computed slice-wise along the reconstructed z
    direction and then averaged. This keeps reported values comparable to common
    MRI papers while preserving physically correct 3D reconstruction first.
    """
    pred_mag = kspace_to_image_magnitude(pred)
    target_mag = kspace_to_image_magnitude(target)
    psnr_vals, ssim_vals = [], []
    for i in range(pred_mag.shape[0]):
        p_vol = pred_mag[i].detach().cpu().numpy()
        t_vol = target_mag[i].detach().cpu().numpy()
        if p_vol.ndim == 2:
            p_vol = p_vol[None, ...]
            t_vol = t_vol[None, ...]
        for z in range(p_vol.shape[0]):
            p, t = p_vol[z], t_vol[z]
            # Normalize by the TARGET's range only, applying the same scale to the
            # prediction so absolute intensity errors are preserved.
            t_min, t_max = t.min(), t.max()
            rng = t_max - t_min + 1e-8
            p, t = (p - t_min) / rng, (t - t_min) / rng
            psnr_vals.append(psnr(t, p, data_range=1))
            ssim_vals.append(ssim(t, p, data_range=1))
    return np.mean(psnr_vals), np.mean(ssim_vals)