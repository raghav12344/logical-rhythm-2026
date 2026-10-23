"""
Data Pipeline and Geometric Augmentation Engine
Realm III: The Leaf Restoration Ritual
Code Sangam 2026 // Logical Rhythm
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class PairedLeafDataset(Dataset):
    """
    Paired dataset for biological leaf super-resolution.
    Synchronously applies orthogonal geometric transformations to both LR and HR
    to ensure zero alignment distortion.
    """
    def __init__(self, lr_dir, hr_dir=None, filenames=None, is_train=True):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.is_train = is_train
        
        if filenames is not None:
            self.filenames = filenames
        else:
            self.filenames = sorted(os.listdir(lr_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, fname)
        
        lr_img = Image.open(lr_path).convert("RGB")
        lr_np = np.asarray(lr_img, dtype=np.float32) / 255.0
        # HWC -> CHW
        lr_t = torch.from_numpy(lr_np).permute(2, 0, 1)

        if self.hr_dir is not None:
            hr_path = os.path.join(self.hr_dir, fname)
            hr_img = Image.open(hr_path).convert("RGB")
            hr_np = np.asarray(hr_img, dtype=np.float32) / 255.0
            hr_t = torch.from_numpy(hr_np).permute(2, 0, 1)

            # Apply identical geometric augmentations to both inputs
            if self.is_train:
                # Random Horizontal Flip
                if random.random() > 0.5:
                    lr_t = torch.flip(lr_t, dims=[2])
                    hr_t = torch.flip(hr_t, dims=[2])
                    
                # Random Vertical Flip
                if random.random() > 0.5:
                    lr_t = torch.flip(lr_t, dims=[1])
                    hr_t = torch.flip(hr_t, dims=[1])
                    
                # Orthogonal 90-degree Rotations (grid-aligned, zero interpolation blur)
                rot_k = random.choice([0, 1, 2, 3])
                if rot_k > 0:
                    lr_t = torch.rot90(lr_t, k=rot_k, dims=[1, 2])
                    hr_t = torch.rot90(hr_t, k=rot_k, dims=[1, 2])

            return lr_t, hr_t, fname
        else:
            return lr_t, fname


def get_dataloaders(data_dir="dataset", batch_size=16, val_split=0.1, seed=42, num_workers=0):
    """
    Creates reproducible, leak-free train and validation data loaders, plus test loader.
    """
    lr_train_dir = os.path.join(data_dir, "train_Low_Resolution")
    hr_train_dir = os.path.join(data_dir, "train_High_Resolution")
    lr_test_dir = os.path.join(data_dir, "test_Low_Resolution")

    all_train_files = sorted(os.listdir(lr_train_dir))
    
    # Deterministic split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(all_train_files))
    rng.shuffle(indices)
    
    split_idx = int(len(indices) * (1.0 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_files = [all_train_files[i] for i in train_indices]
    val_files = [all_train_files[i] for i in val_indices]
    
    train_ds = PairedLeafDataset(lr_train_dir, hr_train_dir, filenames=train_files, is_train=True)
    val_ds = PairedLeafDataset(lr_train_dir, hr_train_dir, filenames=val_files, is_train=False)
    test_ds = PairedLeafDataset(lr_test_dir, hr_dir=None, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Dataset Loaded: {len(train_ds)} Train, {len(val_ds)} Val, {len(test_ds)} Test")
    return train_loader, val_loader, test_loader
