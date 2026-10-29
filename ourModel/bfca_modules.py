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


class FullScaleContextCalibration(nn.Module):
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


class LSKAContext(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FrequencyDetailCalibration(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(skip_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * (1.0 + self.gate(skip))


class FrequencyAwareSkipFusion(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fuse = nn.Sequential(
            ConvBNAct(out_channels + skip_channels, out_channels),
            ConvBNAct(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class FMSA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 16), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 16), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attn(x)


class BEGA(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fusion = FrequencyAwareSkipFusion(in_channels, skip_channels, out_channels)
        self.boundary = nn.Sequential(nn.Conv2d(out_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(x, skip)
        return fused, self.boundary(fused)
