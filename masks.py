# masks.py
import numpy as np
from config import N_PE, N_RO, N_SLICES

def create_circular_base_mask(n_pe=N_PE, n_ro=N_RO):
    """Circular k-space windowing — base for all masks."""
    center_pe, center_ro = n_pe / 2, n_ro / 2
    pe_idx, ro_idx = np.arange(n_pe), np.arange(n_ro)
    PE, RO = np.meshgrid(pe_idx, ro_idx, indexing='ij')
    ellipse = ((PE - center_pe) / (n_pe / 2))**2 + ((RO - center_ro) / (n_ro / 2))**2
    return (ellipse <= 1.0).astype(np.float32)

def create_r1r2_undersampling_mask(n_pe=N_PE, n_ro=N_RO, center_fraction=0.125, acceleration=2, seed=None):
    """Legacy 2D variable-density mask over (ky, kx)."""
    # Use a local generator so we never mutate global NumPy RNG state.
    rng = np.random.default_rng(seed)

    mask = create_circular_base_mask(n_pe, n_ro)
    n_center_pe = max(1, int(round(n_pe * center_fraction)))
    n_center_ro = max(1, int(round(n_ro * center_fraction)))
    center_pe, center_ro = n_pe // 2, n_ro // 2
    pe_start, pe_end = center_pe - n_center_pe // 2, center_pe + n_center_pe // 2
    ro_start, ro_end = center_ro - n_center_ro // 2, center_ro + n_center_ro // 2
    center_locked = np.zeros((n_pe, n_ro), dtype=bool)
    center_locked[pe_start:pe_end, ro_start:ro_end] = True

    pe_idx, ro_idx = np.arange(n_pe), np.arange(n_ro)
    PE, RO = np.meshgrid(pe_idx, ro_idx, indexing='ij')
    norm_dist = np.sqrt(((PE - center_pe) / (n_pe / 2))**2 + ((RO - center_ro) / (n_ro / 2))**2)
    prob_map = np.maximum(0, 1 - norm_dist)**2
    prob_map[center_locked] = 0
    prob_map[mask == 0] = 0
    prob_map = prob_map / prob_map.sum()

    total_in_circle = int(mask.sum())
    center_sampled = int(center_locked[mask == 1].sum())
    target_total = int(total_in_circle / acceleration)
    n_additional = max(0, target_total - center_sampled)

    flat_probs = prob_map.ravel()
    valid_indices = np.where(flat_probs > 0)[0]
    n_additional = min(n_additional, len(valid_indices))
    chosen_flat = rng.choice(len(flat_probs), size=n_additional, replace=False, p=flat_probs)

    final_mask = np.zeros((n_pe, n_ro), dtype=np.float32)
    final_mask[center_locked & (mask == 1)] = 1.0
    final_mask.flat[chosen_flat] = 1.0
    final_mask[mask == 0] = 0.0
    return final_mask


def create_ky_kz_undersampling_mask(n_kz=N_SLICES, n_ky=N_PE, n_kx=N_RO,
                                    center_fraction=0.125, acceleration=2, seed=None):
    """3D Cartesian undersampling mask over phase-encode plane (kz, ky).

    Readout kx is kept fully sampled for every selected (kz, ky) location, which
    is the physically appropriate convention for Cartesian MRI acceleration.
    Returns a mask with shape (kz, ky, kx).
    """
    rng = np.random.default_rng(seed)

    kz_idx, ky_idx = np.arange(n_kz), np.arange(n_ky)
    KZ, KY = np.meshgrid(kz_idx, ky_idx, indexing='ij')
    center_kz, center_ky = n_kz // 2, n_ky // 2

    n_center_kz = max(1, int(round(n_kz * center_fraction)))
    n_center_ky = max(1, int(round(n_ky * center_fraction)))
    kz_start = center_kz - n_center_kz // 2
    kz_end = kz_start + n_center_kz
    ky_start = center_ky - n_center_ky // 2
    ky_end = ky_start + n_center_ky

    center_locked = np.zeros((n_kz, n_ky), dtype=bool)
    center_locked[kz_start:kz_end, ky_start:ky_end] = True

    norm_dist = np.sqrt(
        ((KZ - center_kz) / max(center_kz, 1))**2 +
        ((KY - center_ky) / max(center_ky, 1))**2
    )
    valid = np.ones((n_kz, n_ky), dtype=bool)
    prob_map = np.maximum(0.05, 1 - norm_dist)**2
    prob_map[center_locked] = 0

    final_2d = np.zeros((n_kz, n_ky), dtype=np.float32)
    final_2d[center_locked] = 1.0

    target_total = int(valid.sum() / acceleration)
    n_additional = max(0, target_total - int(final_2d.sum()))
    flat_probs = prob_map.ravel()
    valid_indices = np.where(flat_probs > 0)[0]
    n_additional = min(n_additional, len(valid_indices))

    if n_additional > 0:
        flat_probs = flat_probs / flat_probs.sum()
        chosen_flat = rng.choice(len(flat_probs), size=n_additional, replace=False, p=flat_probs)
        final_2d.flat[chosen_flat] = 1.0

    return np.repeat(final_2d[:, :, None], n_kx, axis=2).astype(np.float32)

def create_fully_sampled_mask(n_pe=N_PE, n_ro=N_RO):
    """Legacy fully sampled 2D mask — circular windowing only."""
    return create_circular_base_mask(n_pe, n_ro)


def create_fully_sampled_3d_mask(n_kz=N_SLICES, n_ky=N_PE, n_kx=N_RO):
    """Fully sampled 3D Cartesian mask."""
    return np.ones((n_kz, n_ky, n_kx), dtype=np.float32)