# data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import DEFAULT_ACCELERATION, DEFAULT_SEED, BASE_PATH
from masks import create_ky_kz_undersampling_mask
from utils import normalize_kspace, load_kspace_optimized
from tqdm import tqdm

# ============================================================
# DATASET LOADING AND SEPARATION
# ============================================================

def load_and_separate_dataset():
    """
    Load all files and separate fully sampled from undersampled using folder structure.
    """
    all_files = []
    
    print(f"Loading dataset from: {BASE_PATH}")
    
    for root, dirs, files in os.walk(BASE_PATH):
        for f in files:
            if not f.endswith('.h5'):
                continue
            full_path = os.path.join(root, f)
            
            # Get relative path from BASE_PATH
            rel_path = os.path.relpath(full_path, BASE_PATH)
            parts = rel_path.split(os.sep)
            
            dataset_id = parts[0] if len(parts) > 0 else "unknown"
            sub_dataset = parts[1] if len(parts) > 1 else "unknown"
            subject = parts[-2] if len(parts) > 1 else "unknown"
            filename = f
            
            # Detect noise_corr
            noise_corr = "on" if "noise_corr_on" in full_path else \
                         "off" if "noise_corr_off" in full_path else "unknown"
            
            # Contrast from filename
            if "T2w" in f:
                contrast = "T2w"
            elif "IR_T1w" in f or "T1w_IR" in f:
                contrast = "IR-T1w"
            elif "T1w" in f or "T1_" in f:
                contrast = "T1w"
            else:
                contrast = "Unknown"
            
            # ── FULLY SAMPLED LOGIC ──
            is_fully_sampled = False
            
            # 1. Dataset 19609495 — everything in repeatability_data is fully sampled
            if dataset_id == "19609495" and sub_dataset == "repeatability_data":
                is_fully_sampled = True
            
            # 2. Dataset 19661402 — repeatability sub-dataset is fully sampled
            if dataset_id == "19661402" and sub_dataset == "repeatability":
                is_fully_sampled = True
            
            # 3. Dataset 19661402 — R1_R2 sub-dataset: _R1 files are fully sampled
            if dataset_id == "19661402" and sub_dataset == "R1_R2":
                if "_R1" in f:
                    is_fully_sampled = True
            
            all_files.append({
                "dataset_id": dataset_id,
                "sub_dataset": sub_dataset,
                "subject": subject,
                "filename": filename,
                "contrast": contrast,
                "noise_corr": noise_corr,
                "full_path": full_path,
                "is_fully_sampled": is_fully_sampled,
            })
    
    df = pd.DataFrame(all_files)
    df['sampling_category'] = df['is_fully_sampled'].map({True: 'Fully Sampled', False: 'Undersampled'})
    
    fully_sampled_df = df[df['is_fully_sampled'] == True].copy().reset_index(drop=True)
    undersampled_df = df[df['is_fully_sampled'] == False].copy().reset_index(drop=True)
    
    print("\n" + "="*60)
    print("DATASET LOADING SUMMARY")
    print("="*60)
    print(f"Total files found        : {len(df)}")
    print(f"  ✅ Fully sampled files : {len(fully_sampled_df)}")
    print(f"  ❌ Undersampled files  : {len(undersampled_df)}")
    print("-"*60)
    print(f"Total unique subjects    : {df['subject'].nunique()}")
    print(f"  ✅ Fully sampled subjects : {fully_sampled_df['subject'].nunique()}")
    print(f"  ❌ Undersampled subjects  : {undersampled_df['subject'].nunique()}")
    print("-"*60)
    
    if len(fully_sampled_df) > 0:
        print(f"Fully sampled subjects: {sorted(fully_sampled_df['subject'].unique())}")
        print(f"Fully sampled datasets: {sorted(fully_sampled_df['dataset_id'].unique())}")
        print(f"Fully sampled sub-datasets: {sorted(fully_sampled_df['sub_dataset'].unique())}")
    
    if len(undersampled_df) > 0:
        print(f"Undersampled subjects: {sorted(undersampled_df['subject'].unique())}")
    
    print("="*60)
    
    return df, fully_sampled_df, undersampled_df
# ============================================================
# DATASET (used for both training folds and validation folds)
# ============================================================
class MRIDataset(Dataset):
    """
    Fully sampled 3D k-space files → synthetic ky-kz undersampling mask applied
    on the fly.

    Each dataset item is now one full volume shaped (kz, ky, kx), not an
    individual kz plane. This enables physically correct 3D FFT reconstruction.

    random_masks=True  → a fresh random mask per volume per access (training:
                         acts as augmentation, prevents mask memorization).
    random_masks=False → one fixed mask per file, pre-generated from `seed`
                         (validation: deterministic, comparable metrics).
    """
    def __init__(self, df, acceleration=DEFAULT_ACCELERATION,
                 seed=DEFAULT_SEED, random_masks=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.acceleration = acceleration
        self.random_masks = random_masks
        self.slices_per_file = []  # retained for backwards-compatible visualization code
        self.total_samples = len(self.df)

        split_name = "train" if random_masks else "val"
        print(f"Loading {len(self.df)} {split_name} files...")
        self.volumes = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Loading {split_name} files"):
            vol, _ = normalize_kspace(load_kspace_optimized(row['full_path']))
            self.volumes.append(vol)
            n_kz = vol.shape[0]
            self.slices_per_file.append(n_kz)

        # Pre-generate fixed masks only for deterministic (validation) mode
        if not random_masks:
            rng = np.random.default_rng(seed)
            self.masks = []
            for vol in self.volumes:
                s = rng.integers(0, 99999)
                self.masks.append(create_ky_kz_undersampling_mask(
                    vol.shape[0], vol.shape[1], vol.shape[2],
                    acceleration=acceleration, seed=int(s)).astype(np.float32))

        print(f"✅ {split_name.capitalize()}: {len(self.df)} 3D volumes loaded")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of bounds")

        file_idx = idx
        full_volume = self.volumes[file_idx]

        if self.random_masks:
            # Fresh random ky-kz mask every access — no seed → non-deterministic.
            mask = create_ky_kz_undersampling_mask(
                full_volume.shape[0], full_volume.shape[1], full_volume.shape[2],
                acceleration=self.acceleration,
            ).astype(np.float32)
        else:
            mask = self.masks[file_idx]
        undersampled = full_volume * mask

        inp = torch.from_numpy(np.stack([undersampled.real, undersampled.imag])).float()
        tgt = torch.from_numpy(np.stack([full_volume.real, full_volume.imag])).float()
        return inp, tgt, torch.from_numpy(mask).float()


# Backwards-compatible alias
ValMRIDataset = MRIDataset