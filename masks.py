# masks.py
import numpy as np
from config import N_PE, N_RO

def create_circular_base_mask(n_pe=N_PE, n_ro=N_RO):
    """Circular k-space windowing — base for all masks."""
    center_pe, center_ro = n_pe / 2, n_ro / 2
    pe_idx, ro_idx = np.arange(n_pe), np.arange(n_ro)
    PE, RO = np.meshgrid(pe_idx, ro_idx, indexing='ij')
    ellipse = ((PE - center_pe) / (n_pe / 2))**2 + ((RO - center_ro) / (n_ro / 2))**2
    return (ellipse <= 1.0).astype(np.float32)

def create_r1r2_undersampling_mask(n_pe=N_PE, n_ro=N_RO, center_fraction=0.125, acceleration=2, seed=None):
    """Variable-density random mask matching R1_R2 acquisition."""
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

def create_fully_sampled_mask(n_pe=N_PE, n_ro=N_RO):
    """Fully sampled mask — circular windowing only."""
    return create_circular_base_mask(n_pe, n_ro)