# data_loader.py
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import N_PE, N_RO, DEFAULT_ACCELERATION, DEFAULT_SEED, BASE_PATH
from masks import create_r1r2_undersampling_mask
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
# TRAINING DATASET
# ============================================================

class TrainMRIDataset(Dataset):
    """Fully sampled files → synthetic mask applied on the fly."""
    def __init__(self, df, acceleration=DEFAULT_ACCELERATION):
        self.df = df.reset_index(drop=True)
        self.acceleration = acceleration
        self.cache = {}
        self.slices_per_file = []
        self.total_samples = 0
        
        print(f"Loading {len(self.df)} training files...")
        
        # ── ADD PROGRESS BAR ──
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading train files"):
            vol = load_kspace_optimized(row['full_path'])
            n_slices = vol.shape[0]
            self.slices_per_file.append(n_slices)
            self.total_samples += n_slices
        
        print(f"✅ Train: {len(self.df)} files, {self.total_samples} slices loaded")
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cumulative = 0
        for i, n_slices in enumerate(self.slices_per_file):
            if idx < cumulative + n_slices:
                file_idx, slice_idx = i, idx - cumulative
                break
            cumulative += n_slices
        else:
            raise IndexError(f"Index {idx} out of bounds")

        if file_idx not in self.cache:
            path = self.df.iloc[file_idx]['full_path']
            vol, _ = normalize_kspace(load_kspace_optimized(path))
            self.cache[file_idx] = vol

        full_slice = self.cache[file_idx][slice_idx]

        # Pad PE to 136
        if full_slice.shape[0] != N_PE:
            pad = N_PE - full_slice.shape[0]
            pad_top, pad_bottom = pad // 2, pad - pad_top
            full_slice = np.pad(full_slice, ((pad_top, pad_bottom), (0, 0)), mode='constant')

        mask = create_r1r2_undersampling_mask(N_PE, N_RO, acceleration=self.acceleration).astype(np.float32)
        undersampled = full_slice * mask

        inp = torch.from_numpy(np.stack([undersampled.real, undersampled.imag])).float()
        tgt = torch.from_numpy(np.stack([full_slice.real, full_slice.imag])).float()
        return inp, tgt, torch.from_numpy(mask).float()


# ============================================================
# VALIDATION DATASET
# ============================================================
class ValMRIDataset(Dataset):
    def __init__(self, df, acceleration=DEFAULT_ACCELERATION, seed=DEFAULT_SEED):
        self.df = df.reset_index(drop=True)
        self.acceleration = acceleration
        self.cache = {}
        self.slices_per_file = []
        self.total_samples = 0

        print(f"Loading {len(self.df)} validation files...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading val files"):
            vol = load_kspace_optimized(row['full_path'])
            n_slices = vol.shape[0]
            self.slices_per_file.append(n_slices)
            self.total_samples += n_slices

        # Pre-generate fixed masks
        rng = np.random.default_rng(seed)
        self.masks = []
        for _ in range(len(self.df)):
            s = rng.integers(0, 99999)
            self.masks.append(create_r1r2_undersampling_mask(N_PE, N_RO, acceleration=acceleration, seed=int(s)).astype(np.float32))

        print(f"✅ Val: {len(self.df)} files, {self.total_samples} slices loaded")
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cumulative = 0
        for i, n_slices in enumerate(self.slices_per_file):
            if idx < cumulative + n_slices:
                file_idx, slice_idx = i, idx - cumulative
                break
            cumulative += n_slices
        else:
            raise IndexError(f"Index {idx} out of bounds")

        if file_idx not in self.cache:
            path = self.df.iloc[file_idx]['full_path']
            vol, _ = normalize_kspace(load_kspace_optimized(path))
            self.cache[file_idx] = vol

        full_slice = self.cache[file_idx][slice_idx]
        if full_slice.shape[0] != N_PE:
            pad = N_PE - full_slice.shape[0]
            pad_top, pad_bottom = pad // 2, pad - pad_top
            full_slice = np.pad(full_slice, ((pad_top, pad_bottom), (0, 0)), mode='constant')

        mask = self.masks[file_idx]
        undersampled = full_slice * mask

        inp = torch.from_numpy(np.stack([undersampled.real, undersampled.imag])).float()
        tgt = torch.from_numpy(np.stack([full_slice.real, full_slice.imag])).float()
        return inp, tgt, torch.from_numpy(mask).float()


# ============================================================
# TEST DATASET
# ============================================================

class TestMRIDataset(Dataset):
    """Test with fixed mask (seed=123) — different from val."""
    def __init__(self, df, acceleration=DEFAULT_ACCELERATION, seed=123):
        self.df = df.reset_index(drop=True)
        self.acceleration = acceleration
        self.cache = {}
        self.slices_per_file = []
        self.total_samples = 0
        for _, row in self.df.iterrows():
            vol = load_kspace_optimized(row['full_path'])
            n_slices = vol.shape[0]
            self.slices_per_file.append(n_slices)
            self.total_samples += n_slices

        rng = np.random.default_rng(seed)
        self.masks = []
        for _ in range(len(self.df)):
            s = rng.integers(0, 99999)
            self.masks.append(create_r1r2_undersampling_mask(N_PE, N_RO, acceleration=acceleration, seed=int(s)).astype(np.float32))

        print(f"TestMRIDataset: {len(self.df)} files, {self.total_samples} slices")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cumulative = 0
        for i, n_slices in enumerate(self.slices_per_file):
            if idx < cumulative + n_slices:
                file_idx, slice_idx = i, idx - cumulative
                break
            cumulative += n_slices
        else:
            raise IndexError(f"Index {idx} out of bounds")

        if file_idx not in self.cache:
            path = self.df.iloc[file_idx]['full_path']
            vol, _ = normalize_kspace(load_kspace_optimized(path))
            self.cache[file_idx] = vol

        full_slice = self.cache[file_idx][slice_idx]
        if full_slice.shape[0] != N_PE:
            pad = N_PE - full_slice.shape[0]
            pad_top, pad_bottom = pad // 2, pad - pad_top
            full_slice = np.pad(full_slice, ((pad_top, pad_bottom), (0, 0)), mode='constant')
        mask = self.masks[file_idx]
        undersampled = full_slice * mask

        inp = torch.from_numpy(np.stack([undersampled.real, undersampled.imag])).float()
        tgt = torch.from_numpy(np.stack([full_slice.real, full_slice.imag])).float()
        return inp, tgt, torch.from_numpy(mask).float()