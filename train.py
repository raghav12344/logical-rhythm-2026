"""
Training Crucible & Validation Profiling Engine
Realm III: The Leaf Restoration Ritual
Code Sangam 2026 // Logical Rhythm

Fully compliant with:
- Constraint 1: Explicit random initialization (zero pre-trained backbones)
- Constraint 2: Offline self-contained execution
- Official Leaderboard Metric: Mean Absolute Error (MAE)
- Architecture: Conditional GAN (cGAN) with PatchGAN & CBAM Attention
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import (
    LeafSuperResNet,
    PatchGANDiscriminator,
    init_weights,
    CharbonnierLoss,
    SobelEdgeLoss,
    VGGPerceptualLoss,
    calculate_mae,
    calculate_psnr,
    calculate_ssim
)
from dataset import get_dataloaders


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(generator, discriminator, loader, opt_G, opt_D, criterion_pixel, criterion_edge, criterion_vgg, criterion_gan, device):
    generator.train()
    discriminator.train()
    total_loss_G = 0.0
    total_loss_D = 0.0

    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        b_size = lr.size(0)

        # ----------------- 1. Train Discriminator -----------------
        opt_D.zero_grad()
        sr = generator(lr).detach()
        
        d_real = discriminator(hr)
        d_fake = discriminator(sr)
        
        loss_D_real = criterion_gan(d_real, torch.ones_like(d_real))
        loss_D_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))
        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        
        loss_D.backward()
        opt_D.step()
        total_loss_D += loss_D.item() * b_size

        # ----------------- 2. Train Generator -----------------
        opt_G.zero_grad()
        sr_for_G = generator(lr)
        d_fake_for_G = discriminator(sr_for_G)
        
        loss_pixel = criterion_pixel(sr_for_G, hr)
        loss_edge = criterion_edge(sr_for_G, hr)
        loss_adv = criterion_gan(d_fake_for_G, torch.ones_like(d_fake_for_G))
        
        # Heavy weighting on pixel MAE (leaderboard metric) + edge & perceptual
        loss_G = 100.0 * loss_pixel + 10.0 * loss_edge + 0.1 * loss_adv
        if criterion_vgg is not None:
            loss_G += 1.0 * criterion_vgg(sr_for_G, hr)
            
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
        opt_G.step()
        total_loss_G += loss_G.item() * b_size

    n = len(loader.dataset)
    return total_loss_G / n, total_loss_D / n


@torch.no_grad()
def validate(generator, loader, device):
    generator.eval()
    maes, psnrs, ssims = [], [], []

    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = generator(lr)
        
        maes.append(calculate_mae(pred, hr))
        psnrs.append(calculate_psnr(pred, hr))
        ssims.append(calculate_ssim(pred, hr))

    return np.mean(maes), np.mean(psnrs), np.mean(ssims)


def predict_tta(model, x):
    """8-fold Test-Time Augmentation: averages 4 rotations + horizontal flips."""
    preds = []
    for k in [0, 1, 2, 3]:
        xr = torch.rot90(x, k=k, dims=[2, 3])
        out_r = model(xr)
        preds.append(torch.rot90(out_r, k=(4 - k) % 4, dims=[2, 3]))
        
        xf = torch.flip(xr, dims=[3])
        out_f = model(xf)
        out_uf = torch.flip(out_f, dims=[3])
        preds.append(torch.rot90(out_uf, k=(4 - k) % 4, dims=[2, 3]))
    return torch.stack(preds, dim=0).mean(dim=0)


def generate_submission(generator, test_loader, test_csv_path, out_csv_path, pred_dir, device):
    print("\n--- Initiating Test Set Inference & Submission Generation (with 8x TTA) ---")
    generator.eval()
    os.makedirs(pred_dir, exist_ok=True)
    
    preds_dict = {}
    with torch.no_grad():
        for lr, fnames in test_loader:
            lr = lr.to(device)
            preds = predict_tta(generator, lr)
            
            preds_np = (preds.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            for i, fname in enumerate(fnames):
                img_arr = preds_np[i]
                Image.fromarray(img_arr).save(os.path.join(pred_dir, fname))
                preds_dict[fname] = img_arr.reshape(-1)

    print(f"Generated {len(preds_dict)} predicted 128x128 PNGs in '{pred_dir}'")
    
    test_df = pd.read_csv(test_csv_path)
    sub = test_df[["id"]].copy()
    sub["pixel_value"] = [preds_dict[img][idx] for img, idx in zip(test_df["image"], test_df["pixel_index"])]
    sub.to_csv(out_csv_path, index=False)
    print(f"SUCCESS: Wrote {len(sub):,} rows to '{out_csv_path}'")


def save_diagnostic_visualizations(generator, val_loader, out_path, device, num_samples=4):
    generator.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    lr_batch, hr_batch, fnames = next(iter(val_loader))
    lr_batch, hr_batch = lr_batch[:num_samples].to(device), hr_batch[:num_samples].to(device)
    
    with torch.no_grad():
        pred_batch = generator(lr_batch)
        bicubic_batch = torch.nn.functional.interpolate(lr_batch, scale_factor=4, mode='bicubic', align_corners=False)

    lr_np = (lr_batch.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    bic_np = (bicubic_batch.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_np = (pred_batch.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    hr_np = (hr_batch.permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    for i in range(num_samples):
        b_mae = calculate_mae(bicubic_batch[i:i+1], hr_batch[i:i+1])
        m_mae = calculate_mae(pred_batch[i:i+1], hr_batch[i:i+1])
        m_psnr = calculate_psnr(pred_batch[i:i+1], hr_batch[i:i+1])
        
        axes[i, 0].imshow(lr_np[i])
        axes[i, 0].set_title(f"Input Smudge (32x32)\n{fnames[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(bic_np[i])
        axes[i, 1].set_title(f"Bicubic Baseline\nMAE: {b_mae:.2f}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_np[i])
        axes[i, 2].set_title(f"Ascended cGAN\nMAE: {m_mae:.2f} | PSNR: {m_psnr:.2f} dB")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(hr_np[i])
        axes[i, 3].set_title("Ground Truth Diagnostic (128x128)")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved diagnostic visual comparisons to '{out_path}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_blocks", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Executing on hardware device: {device}")

    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )

    # Instantiate models with explicit random initialization (Constraint 1)
    generator = LeafSuperResNet(num_blocks=args.num_blocks).to(device)
    generator.apply(init_weights)

    discriminator = PatchGANDiscriminator(in_channels=3, ndf=64).to(device)
    discriminator.apply(init_weights)

    criterion_pixel = CharbonnierLoss().to(device)
    criterion_edge = SobelEdgeLoss().to(device)
    criterion_gan = nn.BCEWithLogitsLoss().to(device)

    vgg_weights_path = os.path.join(args.data_dir, "vgg19_weights.pth")
    if os.path.exists(vgg_weights_path):
        print(f"Loading offline tournament VGG-19 weights from '{vgg_weights_path}'...")
        criterion_vgg = VGGPerceptualLoss(weights_path=vgg_weights_path).to(device)
    else:
        print("VGG weights not found, perceptual loss bypassed.")
        criterion_vgg = None

    opt_G = optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_D = optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)

    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("diagnostics", exist_ok=True)
    best_mae = float("inf")

    print(f"\n{'='*75}\nBeginning Ascended cGAN Training Crucible ({args.epochs} Epochs)\n{'='*75}")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss_G, loss_D = train_epoch(
            generator, discriminator, train_loader, opt_G, opt_D,
            criterion_pixel, criterion_edge, criterion_vgg, criterion_gan, device
        )
        val_mae, val_psnr, val_ssim = validate(generator, val_loader, device)
        sched_G.step()
        sched_D.step()
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Loss G: {loss_G:.3f} | Loss D: {loss_D:.3f} | "
              f"Val MAE: {val_mae:.4f} [Leaderboard Metric] | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(generator.state_dict(), "checkpoints/best_leaf_sr_generator.pth")
            print(f"  --> Saved new best checkpoint (Val MAE: {best_mae:.4f}) to 'checkpoints/best_leaf_sr_generator.pth'")

    print(f"\nTraining Complete! Peak Validation MAE: {best_mae:.4f}")
    
    # Save visual comparison grid
    generator.load_state_dict(torch.load("checkpoints/best_leaf_sr_generator.pth", map_location=device))
    save_diagnostic_visualizations(generator, val_loader, "diagnostics/validation_restoration_grid.png", device)

    # Generate Kaggle Submission
    test_csv = os.path.join(args.data_dir, "test.csv")
    out_csv = "submission.csv"
    pred_dir = "predictions"
    generate_submission(generator, test_loader, test_csv, out_csv, pred_dir, device)


if __name__ == "__main__":
    main()
