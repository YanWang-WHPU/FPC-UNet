from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50

sys.path.append(str(Path(__file__).resolve().parents[1]))
from ourModel.bfca_modules import (
    BEGA,
    FMSA,
    FrequencyAwareSkipFusion,
    FrequencyDetailCalibration,
    FullScaleContextCalibration,
    LSKAContext,
    PrototypeGuidedLogitCalibration,
)  # noqa: E402


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 1, dilation: int = 1):
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(ConvBNAct(in_ch, out_ch), ConvBNAct(out_ch, out_ch))


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        base: int = 32,
        use_lska: bool = False,
        use_fasf: bool = False,
        use_fdc: bool = False,
        use_fscc: bool = False,
        use_pglc: bool = False,
    ):
        super().__init__()
        self.use_fasf = use_fasf
        self.use_fdc = use_fdc
        self.use_fscc = use_fscc
        self.use_pglc = use_pglc
        self.enc1 = DoubleConv(3, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)
        self.context = LSKAContext(base * 16) if use_lska else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        if use_fasf:
            self.dec4 = FrequencyAwareSkipFusion(base * 16, base * 8, base * 8)
            self.dec3 = FrequencyAwareSkipFusion(base * 8, base * 4, base * 4)
            self.dec2 = FrequencyAwareSkipFusion(base * 4, base * 2, base * 2)
            self.dec1 = FrequencyAwareSkipFusion(base * 2, base, base)
        else:
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = DoubleConv(base * 16, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = DoubleConv(base * 2, base)
        if use_fdc:
            self.fdc4 = FrequencyDetailCalibration(base * 8, base * 8)
            self.fdc3 = FrequencyDetailCalibration(base * 4, base * 4)
            self.fdc2 = FrequencyDetailCalibration(base * 2, base * 2)
            self.fdc1 = FrequencyDetailCalibration(base, base)
        if use_fscc:
            self.fscc = FullScaleContextCalibration([base, base * 2, base * 4, base * 8, base * 16], base)
        self.head = nn.Conv2d(base, num_classes, 1)
        if use_pglc:
            self.pglc = PrototypeGuidedLogitCalibration(base, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.context(self.bottleneck(self.pool(e4)))
        if self.use_fasf:
            d4 = self.dec4(b, e4)
            d3 = self.dec3(d4, e3)
            d2 = self.dec2(d3, e2)
            d1 = self.dec1(d2, e1)
        else:
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            if self.use_fdc:
                d4 = self.fdc4(d4, e4)
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            if self.use_fdc:
                d3 = self.fdc3(d3, e3)
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            if self.use_fdc:
                d2 = self.fdc2(d2, e2)
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            if self.use_fdc:
                d1 = self.fdc1(d1, e1)
        if self.use_fscc:
            d1 = self.fscc(d1, [e1, e2, e3, e4, b])
        logits = self.head(d1)
        if self.use_pglc:
            logits = self.pglc(d1, logits)
        return logits


class PSPNetLite(nn.Module):
    def __init__(self, num_classes: int = 4, encoder: str = "resnet34", channels: int = 256):
        super().__init__()
        self.encoder = timm.create_model(encoder, features_only=True, pretrained=False, in_chans=3)
        enc_ch = self.encoder.feature_info.channels()[-1]
        self.project = ConvBNAct(enc_ch * 5, channels, kernel_size=1)
        self.head = nn.Sequential(ConvBNAct(channels, channels // 2), nn.Conv2d(channels // 2, num_classes, 1))
        self.pool_sizes = (1, 2, 3, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feat = self.encoder(x)[-1]
        pooled = [feat]
        for ps in self.pool_sizes:
            y = F.adaptive_avg_pool2d(feat, ps)
            y = F.interpolate(y, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            pooled.append(y)
        out = self.head(self.project(torch.cat(pooled, dim=1)))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class FeatureDecoder(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        decoder_channels: int = 128,
        mode: str = "unetformer",
        use_bega: bool = False,
        use_fmsa: bool = False,
        use_fscc: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.use_bega = use_bega
        self.use_fmsa = use_fmsa
        self.use_fscc = use_fscc
        self.in_channels = in_channels
        self.proj = nn.ModuleList([ConvBNAct(ch, decoder_channels, kernel_size=1) for ch in in_channels])
        self.fuse = nn.ModuleList()
        self.bega = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            if use_bega:
                self.bega.append(BEGA(decoder_channels, decoder_channels, decoder_channels))
            else:
                self.fuse.append(nn.Sequential(ConvBNAct(decoder_channels * 2, decoder_channels), ConvBNAct(decoder_channels, decoder_channels)))
        dilations = (1, 2, 3) if mode == "dcswin" else (1, 3, 5)
        self.context = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels, dilation=dilations[0]),
            ConvBNAct(decoder_channels, decoder_channels, dilation=dilations[1]),
            ConvBNAct(decoder_channels, decoder_channels, dilation=dilations[2]),
        )
        self.fmsa = FMSA(decoder_channels) if use_fmsa else nn.Identity()
        self.fscc = FullScaleContextCalibration([decoder_channels] * len(in_channels), decoder_channels) if use_fscc else nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats = [
            feat.permute(0, 3, 1, 2).contiguous() if feat.ndim == 4 and feat.shape[-1] == ch and feat.shape[1] != ch else feat
            for feat, ch in zip(feats, self.in_channels)
        ]
        feats = [proj(feat) for proj, feat in zip(self.proj, feats)]
        x = self.context(feats[-1])
        boundaries: list[torch.Tensor] = []
        fuse_idx = 0
        bega_idx = 0
        for skip in reversed(feats[:-1]):
            if self.use_bega:
                x, boundary = self.bega[bega_idx](x, skip)
                boundaries.append(boundary)
                bega_idx += 1
            else:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = self.fuse[fuse_idx](torch.cat([x, skip], dim=1))
                fuse_idx += 1
        if self.use_fscc:
            x = self.fscc(x, feats)
        return self.fmsa(x), boundaries


class EncoderDecoderSeg(nn.Module):
    def __init__(
        self,
        encoder: str,
        num_classes: int = 4,
        decoder_channels: int = 128,
        mode: str = "unetformer",
        use_bega: bool = False,
        use_fmsa: bool = False,
        use_fscc: bool = False,
        use_pglc: bool = False,
    ):
        super().__init__()
        kwargs = {"features_only": True, "pretrained": False, "in_chans": 3}
        if encoder.startswith("swin"):
            kwargs["img_size"] = 512
        self.encoder = timm.create_model(encoder, **kwargs)
        in_channels = self.encoder.feature_info.channels()
        self.decoder = FeatureDecoder(in_channels, decoder_channels, mode=mode, use_bega=use_bega, use_fmsa=use_fmsa, use_fscc=use_fscc)
        self.head = nn.Sequential(ConvBNAct(decoder_channels, decoder_channels // 2), nn.Conv2d(decoder_channels // 2, num_classes, 1))
        self.pglc = PrototypeGuidedLogitCalibration(decoder_channels, num_classes) if use_pglc else None

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        size = x.shape[-2:]
        feat, boundaries = self.decoder(self.encoder(x))
        out = self.head(feat)
        if self.pglc is not None:
            out = self.pglc(feat, out)
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        if return_aux:
            return out, {"boundaries": boundaries}
        return out


class SegFormerMLPDecoder(nn.Module):
    def __init__(self, in_channels: list[int], embed_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.ModuleList([nn.Conv2d(ch, embed_dim, 1) for ch in in_channels])
        self.fuse = nn.Sequential(
            ConvBNAct(embed_dim * len(in_channels), embed_dim, kernel_size=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        feats = [
            feat.permute(0, 3, 1, 2).contiguous() if feat.ndim == 4 and feat.shape[-1] == ch and feat.shape[1] != ch else feat
            for feat, ch in zip(feats, self.in_channels)
        ]
        size = feats[0].shape[-2:]
        projected = [
            F.interpolate(proj(feat), size=size, mode="bilinear", align_corners=False)
            if feat.shape[-2:] != size
            else proj(feat)
            for proj, feat in zip(self.proj, feats)
        ]
        return self.fuse(torch.cat(projected, dim=1))


class SegFormerStyleSeg(nn.Module):
    def __init__(self, encoder: str = "swin_tiny_patch4_window7_224", num_classes: int = 4, embed_dim: int = 256):
        super().__init__()
        kwargs = {"features_only": True, "pretrained": False, "in_chans": 3}
        if encoder.startswith("swin"):
            kwargs["img_size"] = 512
        self.encoder = timm.create_model(encoder, **kwargs)
        in_channels = self.encoder.feature_info.channels()
        self.decoder = SegFormerMLPDecoder(in_channels, embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        out = self.decoder(self.encoder(x))
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


class FCNWrapper(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.model = fcn_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def build_model(name: str, num_classes: int = 4) -> nn.Module:
    key = name.lower()
    if key == "unet":
        return UNet(num_classes=num_classes, base=32)
    if key == "unet_lska":
        return UNet(num_classes=num_classes, base=32, use_lska=True)
    if key == "unet_fasf":
        return UNet(num_classes=num_classes, base=32, use_fasf=True)
    if key == "unet_lska_fasf":
        return UNet(num_classes=num_classes, base=32, use_lska=True, use_fasf=True)
    if key == "unet_fdc":
        return UNet(num_classes=num_classes, base=32, use_fdc=True)
    if key == "unet_lska_fdc":
        return UNet(num_classes=num_classes, base=32, use_lska=True, use_fdc=True)
    if key == "unet_fscc":
        return UNet(num_classes=num_classes, base=32, use_fscc=True)
    if key == "unet_lska_fscc":
        return UNet(num_classes=num_classes, base=32, use_lska=True, use_fscc=True)
    if key == "unet_pglc":
        return UNet(num_classes=num_classes, base=32, use_pglc=True)
    if key == "unet_fscc_pglc":
        return UNet(num_classes=num_classes, base=32, use_fscc=True, use_pglc=True)
    if key == "fpc_unet":
        return UNet(num_classes=num_classes, base=32, use_fscc=True, use_pglc=True)
    if key == "deeplabv3":
        return DeepLabV3Wrapper(num_classes=num_classes)
    if key == "fcn":
        return FCNWrapper(num_classes=num_classes)
    if key == "pspnet":
        return PSPNetLite(num_classes=num_classes)
    if key == "unetformer":
        return EncoderDecoderSeg("resnet34", num_classes=num_classes, decoder_channels=128, mode="unetformer")
    if key == "dcswin":
        return EncoderDecoderSeg("swin_tiny_patch4_window7_224", num_classes=num_classes, decoder_channels=128, mode="dcswin")
    if key == "segformer":
        return SegFormerStyleSeg(num_classes=num_classes)
    if key == "bfca_unetformer":
        return EncoderDecoderSeg("resnet34", num_classes=num_classes, decoder_channels=128, mode="unetformer", use_bega=True, use_fmsa=True)
    if key == "bfca_bega":
        return EncoderDecoderSeg("resnet34", num_classes=num_classes, decoder_channels=128, mode="unetformer", use_bega=True, use_fmsa=False)
    if key == "bfca_fmsa":
        return EncoderDecoderSeg("resnet34", num_classes=num_classes, decoder_channels=128, mode="unetformer", use_bega=False, use_fmsa=True)
    if key == "bfca_dcswin":
        return EncoderDecoderSeg("swin_tiny_patch4_window7_224", num_classes=num_classes, decoder_channels=128, mode="dcswin", use_bega=True, use_fmsa=True)
    if key == "fpc_dcswin":
        return EncoderDecoderSeg("swin_tiny_patch4_window7_224", num_classes=num_classes, decoder_channels=128, mode="dcswin", use_fscc=True, use_pglc=True)
    raise ValueError(f"Unknown model: {name}")
