"""
Standalone Inference & Test Deliverables Generator
Realm III: The Leaf Restoration Ritual
Code Sangam 2026 // Logical Rhythm
"""

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch

from models import LeafSuperResNet
from dataset import PairedLeafDataset
from torch.utils.data import DataLoader


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_leaf_sr_generator.pth")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--pred_dir", type=str, default="predictions")
    parser.add_argument("--out_csv", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_blocks", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = LeafSuperResNet(num_blocks=args.num_blocks).to(device)
    if os.path.exists(args.checkpoint):
        print(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"WARNING: Checkpoint {args.checkpoint} not found! Using raw model.")
    model.eval()

    # Load test data
    lr_test_dir = os.path.join(args.data_dir, "test_Low_Resolution")
    test_ds = PairedLeafDataset(lr_test_dir, hr_dir=None, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    os.makedirs(args.pred_dir, exist_ok=True)
    preds_dict = {}

    print(f"Generating super-resolved images for {len(test_ds)} test smudges with 8x TTA...")
    with torch.no_grad():
        for lr, fnames in test_loader:
            lr = lr.to(device)
            preds = predict_tta(model, lr)
            preds_np = (preds.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            for i, fname in enumerate(fnames):
                arr = preds_np[i]
                Image.fromarray(arr).save(os.path.join(args.pred_dir, fname))
                preds_dict[fname] = arr.reshape(-1)

    print(f"Saved {len(preds_dict)} predictions in '{args.pred_dir}'.")

    # Format submission CSV
    test_csv_path = os.path.join(args.data_dir, "test.csv")
    print(f"Loading {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)

    sub = test_df[["id"]].copy()
    sub["pixel_value"] = [preds_dict[img][idx] for img, idx in zip(test_df["image"], test_df["pixel_index"])]
    sub.to_csv(args.out_csv, index=False)
    print(f"Submission written successfully to '{args.out_csv}' ({len(sub)} rows).")


if __name__ == "__main__":
    main()
