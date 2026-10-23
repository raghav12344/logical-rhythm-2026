"""
Exploratory Data Analysis and Biological Degradation Profiling
Realm III: The Leaf Restoration Ritual
Code Sangam 2026 // Logical Rhythm
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.makedirs("eda_outputs", exist_ok=True)

data_dir = "dataset"
hr_dir = os.path.join(data_dir, "train_High_Resolution")
lr_dir = os.path.join(data_dir, "train_Low_Resolution")
test_lr_dir = os.path.join(data_dir, "test_Low_Resolution")

hr_files = sorted(os.listdir(hr_dir))
lr_files = sorted(os.listdir(lr_dir))
test_files = sorted(os.listdir(test_lr_dir))

print(f"Total Paired Train Samples: {len(hr_files)}")
print(f"Total Test Samples: {len(test_files)}")

# 1. Dataset-Wide Statistics
r_means_lr, g_means_lr, b_means_lr = [], [], []
r_means_hr, g_means_hr, b_means_hr = [], [], []

sample_subset = hr_files[:200]
bicubic_psnrs = []

for f in sample_subset:
    hr_img = np.asarray(Image.open(os.path.join(hr_dir, f)).convert("RGB"), dtype=np.float32)
    lr_pil = Image.open(os.path.join(lr_dir, f)).convert("RGB")
    lr_img = np.asarray(lr_pil, dtype=np.float32)
    
    bic_up = np.asarray(lr_pil.resize((128, 128), Image.Resampling.BICUBIC), dtype=np.float32)
    mse = np.mean((hr_img - bic_up) ** 2)
    psnr = 10 * math.log10(255.0 ** 2 / (mse + 1e-10))
    bicubic_psnrs.append(psnr)
    
    r_means_lr.append(lr_img[:, :, 0].mean())
    g_means_lr.append(lr_img[:, :, 1].mean())
    b_means_lr.append(lr_img[:, :, 2].mean())
    
    r_means_hr.append(hr_img[:, :, 0].mean())
    g_means_hr.append(hr_img[:, :, 1].mean())
    b_means_hr.append(hr_img[:, :, 2].mean())

print(f"Mean Bicubic PSNR (200 train images): {np.mean(bicubic_psnrs):.2f} dB (std: {np.std(bicubic_psnrs):.2f})")
print(f"LR RGB Means: R={np.mean(r_means_lr):.2f}, G={np.mean(g_means_lr):.2f}, B={np.mean(b_means_lr):.2f}")
print(f"HR RGB Means: R={np.mean(r_means_hr):.2f}, G={np.mean(g_means_hr):.2f}, B={np.mean(b_means_hr):.2f}")

# Plot 1: Visual Comparison & Error Residual Maps
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

sample_indices = [0, 15, 42]
for row_idx, idx in enumerate(sample_indices):
    fname = hr_files[idx]
    hr_img = Image.open(os.path.join(hr_dir, fname)).convert("RGB")
    lr_img = Image.open(os.path.join(lr_dir, fname)).convert("RGB")
    bic_up = lr_img.resize((128, 128), Image.Resampling.BICUBIC)
    
    hr_arr = np.asarray(hr_img)
    lr_arr = np.asarray(lr_img)
    bic_arr = np.asarray(bic_up)
    
    diff_map = np.abs(hr_arr.astype(np.float32) - bic_arr.astype(np.float32)).mean(axis=2)
    
    axes[row_idx, 0].imshow(lr_arr)
    axes[row_idx, 0].set_title(f"Degraded LR (32x32)\n{fname}")
    axes[row_idx, 0].axis("off")
    
    axes[row_idx, 1].imshow(bic_arr)
    axes[row_idx, 1].set_title(f"Bicubic 4x Baseline (128x128)\nPSNR: {bicubic_psnrs[idx]:.1f} dB")
    axes[row_idx, 1].axis("off")
    
    im_err = axes[row_idx, 2].imshow(diff_map, cmap="inferno", vmin=0, vmax=60)
    axes[row_idx, 2].set_title("Bicubic Residual Error\n(Venation & Lesion Loss)")
    axes[row_idx, 2].axis("off")
    plt.colorbar(im_err, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)
    
    axes[row_idx, 3].imshow(hr_arr)
    axes[row_idx, 3].set_title("Ground Truth HR (128x128)\nTarget Diagnostic")
    axes[row_idx, 3].axis("off")

plt.tight_layout()
plt.savefig("eda_outputs/degradation_profiling.png", dpi=200)
plt.close()
print("Saved eda_outputs/degradation_profiling.png")

# Plot 2: Frequency Domain Analysis (2D Fast Fourier Transform)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

sample_f = hr_files[0]
hr_img = np.asarray(Image.open(os.path.join(hr_dir, sample_f)).convert("L"), dtype=np.float32)
lr_img = np.asarray(Image.open(os.path.join(lr_dir, sample_f)).convert("L"), dtype=np.float32)
bic_img = np.asarray(Image.open(os.path.join(lr_dir, sample_f)).convert("L").resize((128, 128), Image.Resampling.BICUBIC), dtype=np.float32)

fft_hr = np.fft.fftshift(np.fft.fft2(hr_img))
mag_hr = 20 * np.log(np.abs(fft_hr) + 1)

fft_bic = np.fft.fftshift(np.fft.fft2(bic_img))
mag_bic = 20 * np.log(np.abs(fft_bic) + 1)

axes[0, 0].imshow(lr_img, cmap='gray')
axes[0, 0].set_title("LR Input (32x32)")
axes[0, 0].axis('off')

axes[0, 1].imshow(bic_img, cmap='gray')
axes[0, 1].set_title("Bicubic Interpolation (128x128)")
axes[0, 1].axis('off')

axes[0, 2].imshow(hr_img, cmap='gray')
axes[0, 2].set_title("HR Ground Truth (128x128)")
axes[0, 2].axis('off')

axes[1, 0].axis('off')
axes[1, 0].text(0.1, 0.5, "Frequency Domain Spectrum\n\n- Low-pass filtering causes severe\n  high-frequency energy drop.\n- Model must hallucinate\n  biologically plausible high\n  frequencies for plant venation.", fontsize=11)

im_bic_fft = axes[1, 1].imshow(mag_bic, cmap='magma')
axes[1, 1].set_title("Bicubic 2D FFT Magnitude\n(Severe High-Freq Attenuation)")
axes[1, 1].axis('off')
plt.colorbar(im_bic_fft, ax=axes[1, 1], fraction=0.046, pad=0.04)

im_hr_fft = axes[1, 2].imshow(mag_hr, cmap='magma')
axes[1, 2].set_title("HR 2D FFT Magnitude\n(Rich High-Freq Radial Energy)")
axes[1, 2].axis('off')
plt.colorbar(im_hr_fft, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("eda_outputs/fft_frequency_spectrum.png", dpi=200)
plt.close()
print("Saved eda_outputs/fft_frequency_spectrum.png")
