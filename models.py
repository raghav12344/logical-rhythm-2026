"""
Ascended Neural Tabular & Vision Architectures for Realm III: The Leaf Restoration Ritual
Code Sangam 2026 // Logical Rhythm

Implements:
- CBAM (Channel & Spatial Attention Modules) for biological feature salience
- Deep Residual Generator with cascaded PixelShuffle upsampling and Global Bicubic Skip Anchor
- Multi-objective Composite Loss: Charbonnier Loss + Sobel Edge Loss + VGG19 Perceptual Loss
- Vectorized PSNR & SSIM metrics for validation profiling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torchvision.models as models


# ==========================================
# 0. Deterministic Random Initialization (Constraint 1)
# ==========================================
def init_weights(m):
    """
    Strictly satisfies Competition Constraint 1:
    'You must initialize your networks randomly. The use of pre-trained weights
    to solve the core task is strictly prohibited.'
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


# ==========================================
# 1. Attention Modules (CBAM)
# ==========================================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module: Exploits inter-channel relationships of biological features
    (e.g., distinguishing healthy chlorophyll green bands from necrotic brown/yellow shades).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.PReLU(),
            nn.Linear(mid_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module: Exploits spatial relationships to focus feature representation
    on fine plant venation networks and localized necrotic lesion boundaries.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAMResidualBlock(nn.Module):
    """
    Residual Block reinforced with Channel and Spatial Attention (CBAM)
    with a residual scaling factor for stable deep gradient backpropagation.
    """
    def __init__(self, channels=64, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = self.act(self.conv1(x))
        residual = self.conv2(residual)
        residual = self.ca(residual)
        residual = self.sa(residual)
        return x + residual * self.res_scale


# ==========================================
# 2. Ascended Generator: LeafSuperResNet
# ==========================================
class LeafSuperResNet(nn.Module):
    """
    Ascended Super-Resolution Generator for 4x Blind Super-Resolution (32x32 -> 128x128).
    
    Architecture Highlights:
    1. Shallow Feature Extraction: 3x3 Conv mapping 3 -> 64 feature maps.
    2. Deep Residual Attention Trunk: Stack of CBAM residual blocks.
    3. Global Residual Trunk Connection: Stabilizes deep representations.
    4. Cascaded Sub-Pixel Upsampling: Two sequential PixelShuffle(2) stages
       (32x32 -> 64x64 -> 128x128) eliminating checkerboard deconvolution artifacts.
    5. Global Bicubic Anchor: Learns strictly the high-frequency residual Delta = (HR - Bicubic(LR)).
    """
    def __init__(self, num_blocks=8, num_features=64):
        super().__init__()
        self.shallow = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[CBAMResidualBlock(num_features) for _ in range(num_blocks)])
        self.conv_trunk = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Cascaded PixelShuffle Upsampling (2x -> 2x = 4x total)
        self.up1 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.reconstruct = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Global Bicubic Anchor prevents color drift and guarantees baseline fidelity
        bicubic = F.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        
        feat = self.shallow(x)
        res = self.body(feat)
        res = self.conv_trunk(res) + feat
        
        up = self.up1(res)
        up = self.up2(up)
        delta = self.reconstruct(up)
        
        # Output bounded to valid normalized RGB [0.0, 1.0]
        return torch.clamp(bicubic + delta, 0.0, 1.0)


# ==========================================
# 3. Ascended PatchGAN Discriminator (cGAN / SRGAN)
# ==========================================
class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization for stabilized adversarial training.
    Enforces sharp biological realism across 14x14 local receptive field patches.
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 4. Multi-Objective Composite Loss Functions
# ==========================================
class CharbonnierLoss(nn.Module):
    """
    Differentiable smooth L1 loss approximation:
    L(y, y_hat) = sqrt((y - y_hat)^2 + eps^2)
    Directly optimizes pixel MAE while remaining strictly differentiable everywhere.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps2)
        return torch.mean(loss)


class SobelEdgeLoss(nn.Module):
    """
    Sobel Gradient Loss enforcing sharp biological contours (chlorosis boundaries,
    necrotic lesion margins, and fine vein structure).
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        
        gx_p = F.conv2d(pred_gray, self.sobel_x, padding=1)
        gy_p = F.conv2d(pred_gray, self.sobel_y, padding=1)
        gx_t = F.conv2d(target_gray, self.sobel_x, padding=1)
        gy_t = F.conv2d(target_gray, self.sobel_y, padding=1)
        
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class VGGPerceptualLoss(nn.Module):
    """
    Feature-level perceptual loss using offline VGG-19 weights provided by tournament organizers.
    Extracts features up to relu3_3 (layer 16) to guide semantic plant texture fidelity.
    """
    def __init__(self, weights_path='dataset/vgg19_weights.pth'):
        super().__init__()
        vgg = models.vgg19(weights=None)
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        vgg.load_state_dict(weights)
        
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
            
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        f_pred = self.features(pred_norm)
        f_target = self.features(target_norm)
        return F.l1_loss(f_pred, f_target)


# ==========================================
# 5. Evaluation Metrics: MAE, PSNR & SSIM
# ==========================================
def calculate_mae(pred, target):
    """
    Direct Kaggle Leaderboard Metric:
    Mean Absolute Error (MAE) computed over [0, 255] uint8 pixel values.
    """
    pred_255 = pred * 255.0
    target_255 = target * 255.0
    return F.l1_loss(pred_255, target_255).item()


def calculate_psnr(pred, target):
    """
    Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) for inputs in range [0.0, 1.0].
    """
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr.mean().item()


def calculate_ssim(img1, img2, window_size=11):
    """
    Structural Similarity Index (SSIM) for inputs in range [0.0, 1.0].
    """
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(img1.size(1), 1, window_size, window_size).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.size(1)) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()
