from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
        dilation: int = 1,
    ):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            ConvBNAct(in_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )


class FullScaleContextCalibration(nn.Module):
    """Full-scale context calibration used by FPC-UNet.

    It projects multi-scale encoder/bottleneck features to the decoder feature
    resolution, selects scale-wise weights, and adds a residual context term to
    the final decoder feature.
    """

    def __init__(self, channels: list[int], out_channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or out_channels
        self.projects = nn.ModuleList([ConvBNAct(ch, hidden, kernel_size=1) for ch in channels])
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * len(channels), max(hidden // 2, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(hidden // 2, 16), len(channels), 1),
        )
        self.out = ConvBNAct(hidden, out_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, target: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
        size = target.shape[-2:]
        projected = [
            F.interpolate(project(feat), size=size, mode="bilinear", align_corners=False)
            if feat.shape[-2:] != size
            else project(feat)
            for project, feat in zip(self.projects, features)
        ]
        weights = torch.softmax(self.selector(torch.cat(projected, dim=1)), dim=1)
        fused = torch.zeros_like(projected[0])
        for idx, feat in enumerate(projected):
            fused = fused + feat * weights[:, idx : idx + 1]
        return target + self.scale * self.out(fused)


class PrototypeGuidedLogitCalibration(nn.Module):
    """Prototype-guided class-wise correction for raw segmentation logits."""

    def __init__(self, in_channels: int, num_classes: int, proto_channels: int = 32):
        super().__init__()
        self.embed = ConvBNAct(in_channels, proto_channels, kernel_size=1)
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_channels) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.class_scale = nn.Parameter(torch.zeros(num_classes))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, feature: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        embed = F.normalize(self.embed(feature), dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.einsum("bchw,nc->bnhw", embed, prototypes)
        temperature = F.softplus(self.temperature) + 1.0
        correction = torch.tanh(similarity * temperature)
        correction = correction * self.class_scale[None, :, None, None] + self.class_bias[None, :, None, None]
        return logits + correction


class FPCUNet(nn.Module):
    """FPC-UNet extracted from experiment/models.py.

    Equivalent to:
        UNet(num_classes=num_classes, base=32, use_fscc=True, use_pglc=True)
    in the original project.
    """

    def __init__(self, num_classes: int = 4, base: int = 32, in_channels: int = 3):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.fscc = FullScaleContextCalibration(
            [base, base * 2, base * 4, base * 8, base * 16],
            base,
        )
        self.head = nn.Conv2d(base, num_classes, 1)
        self.pglc = PrototypeGuidedLogitCalibration(base, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        d1 = self.fscc(d1, [e1, e2, e3, e4, b])
        logits = self.head(d1)
        logits = self.pglc(d1, logits)
        return logits


def build_fpc_unet(num_classes: int = 4, base: int = 32, in_channels: int = 3) -> FPCUNet:
    return FPCUNet(num_classes=num_classes, base=base, in_channels=in_channels)


__all__ = [
    "ConvBNAct",
    "DoubleConv",
    "FullScaleContextCalibration",
    "PrototypeGuidedLogitCalibration",
    "FPCUNet",
    "build_fpc_unet",
]
