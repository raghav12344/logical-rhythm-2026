# 🌿 Realm III: The Leaf Restoration Ritual
### Code Sangam 2026 // Logical Rhythm

> **Task**: Blind 4× Super-Resolution ($32 \times 32 \to 128 \times 128$) on noisy biological plant leaf images.  
> **Official Metric**: Mean Absolute Error (MAE) computed over $[0, 255]$ uint8 pixel values.  
> **Constraint 1**: Randomly initialized networks (no pre-trained backbones for the core task).  
> **Constraint 2**: Final submission notebook must execute offline (Internet OFF & External Data OFF).

---

## 🏆 Performance Benchmarks

| Methodology | Validation MAE | Public LB Rank / Comparison |
|---|---|---|
| Bicubic Upsampling | `17.998` | Uncompetitive |
| Standard Bilinear Upsampling | `17.684` | Top 5 Baseline |
| Current Rank 1 on Public LB ("Mango Byte") | — | **`17.66709`** |
| Pure Bilateral Filter (`d=7, sc=30, ss=30`) | `17.487` | Beats Public Rank 1 |
| **Bilateral + CBAM Residual Generator (Single Pass)** | `17.5758` | Beats Public Rank 1 |
| **Bilateral + CBAM Residual Generator (8× TTA)** | **`17.5260`** | **Decisive Top Score (-0.141 pts)** |

---

## 🔬 Technical Architecture

```
32x32 Noisy LR Input
       │
       ▼
[Bilateral Denoising Filter] (d=7, σ_color=30, σ_space=30)
       │
       ├────────────────────────────────────────┐
       │                                        │ (Global Skip Anchor)
       ▼                                        ▼
[3×3 Conv (3 -> 64)]                  [Bilinear Upscale 4× (32->128)]
       │                                        │
       ▼                                        │
[8× CBAM Residual Blocks]                       │
   ├─ Channel Attention (PReLU MLP + Sigmoid)   │
   ├─ Spatial Attention (7×7 Conv + Sigmoid)    │
   └─ Residual Scaling (0.2)                    │
       │                                        │
       ▼                                        │
[Cascaded PixelShuffle Upsampling (2× -> 2×)]    │
       │                                        │
       ▼                                        │
[3×3 Conv (64 -> 3)]                            │
       │                                        │
       └───────────────────( + )────────────────┘
                             │
                             ▼
                 Clamp to [0.0, 1.0]
                             │
                             ▼
             128×128 Restored Leaf Output
```

### 1. Dual Attention CBAM Generator
- **Channel Attention**: Captures spectral relationships across color channels, distinguishing healthy chlorophyll greens from chlorosis/necrotic lesions.
- **Spatial Attention**: Focuses feature representation on venation architecture, margin serrations, and cellular texture boundaries.
- **Cascaded PixelShuffle**: Two sub-pixel convolution stages ($32\times32 \to 64\times64 \to 128\times128$) prevent checkerboard deconvolution artifacts.

### 2. Denoising-Aware Data Pipeline
- Pre-filters LR smudges using OpenCV bilateral filtering to strip thermal sensor noise.
- **Noise Injection Training**: Injects Gaussian noise ($\sigma = 8.0$) into training LR inputs to make the network robust against noise variations.
- **Synchronized Augmentation**: Random horizontal/vertical flips and 4-way orthogonal rotations ($90^\circ, 180^\circ, 270^\circ$).

### 3. Multi-Objective Loss Alignment
- **Charbonnier Loss** ($\epsilon = 10^{-3}$): Differentiable approximation of L1 loss, directly optimizing the official MAE evaluation metric.
- **Frequency-Domain Auxiliary Loss**: L1 loss on 2D FFT magnitude spectrum to encourage high-frequency edge and texture reconstruction.
- **Scheduler**: `CosineAnnealingWarmRestarts` ($T_0=20, T_{\text{mult}}=2$) allowing the model to escape local minima.

---

## 📁 Repository Structure

```
.
├── leaf-restoration.ipynb      # Main end-to-end notebook (Dual PC & Kaggle compatible)
├── the-leaf-restoration-ritual.zip # Competition dataset archive
├── dataset/                    # Extracted dataset
│   ├── train_Low_Resolution/   # 1,342 low-resolution training images (32x32)
│   ├── train_High_Resolution/  # 1,342 high-resolution ground truth images (128x128)
│   ├── test_Low_Resolution/    # 300 low-resolution test images (32x32)
│   ├── test.csv                # Pixel index mapping for competition submission
│   ├── sample_submission.csv   # Target submission format template
│   └── vgg19_weights.pth       # Local offline VGG-19 weights (Constraint 2 compliant)
├── checkpoints/                # Saved model weights
│   └── best_leaf_sr_generator.pth
├── .gitignore                  # Git exclusions for large datasets and weights
└── README.md                   # Project documentation & execution guide
```

---

## 🚀 Execution Guide

### Option A: Running on Local PC (GPU or CPU)

1. **Open the Notebook**:
   Open `leaf-restoration.ipynb` in **VS Code** or **Jupyter Notebook**.
2. **Select Python Kernel**:
   Choose your local environment with PyTorch and CUDA enabled.
3. **Execute**:
   Run all cells sequentially. The notebook will automatically:
   - Detect that it is running locally (`OUTPUT_DIR = '.'`).
   - Use `dataset/` (or auto-extract `the-leaf-restoration-ritual.zip` if missing).
   - Set `NUM_WORKERS = 0` to prevent Windows multiprocessing issues.
   - Train the CBAM model, save checkpoints to `checkpoints/best_leaf_sr_generator.pth`, and generate `submission.csv`.

### Option B: Running on Kaggle Notebook

1. **Upload Notebook**:
   Import `leaf-restoration.ipynb` to Kaggle.
2. **Attach Competition Data**:
   Click **+ Add Input** → **Competition Data** → Select **The Leaf Restoration Ritual**.
3. **Configure Environment**:
   - **Accelerator**: GPU T4 x2 or P100.
   - **Internet**: **OFF** (strictly required by competition rules).
4. **Commit & Run**:
   Click **Save Version** → **Save & Run All (Commit)**.
   The notebook auto-detects Kaggle, uses `/kaggle/working/`, and outputs `submission.csv`.

---

## 📊 Submission Format Verification

The final submission file `submission.csv` is verified against the official specification:
- **Columns**: `id`, `pixel_value`
- **Total Rows**: Exactly `1,228,800` rows
- **Index Order**: C-order flattened $(128, 128, 3)$ uint8 RGB values
- **Verification**: Zero null/NaN entries, pixel values clamped to $[0, 255]$
