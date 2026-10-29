"""Predict a class-ID mask for one RGB image using a local FPC-UNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from extracted_fpc_unet import build_fpc_unet


@torch.inference_mode()
def predict(checkpoint: Path, image_path: Path, output: Path, device: str, num_classes: int = 4, base: int = 32) -> None:
    if not 1 <= num_classes <= 256:
        raise ValueError("The PNG class-ID output supports 1 to 256 classes.")
    if output.suffix.lower() != ".png":
        raise ValueError("Use a .png output to preserve integer class IDs.")
    if output.resolve() in (image_path.resolve(), checkpoint.resolve()):
        raise ValueError("Output must not overwrite the input image or checkpoint.")
    model = build_fpc_unet(num_classes=num_classes, base=base).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    with Image.open(image_path) as source:
        x = TF.to_tensor(source.convert("RGB"))
    x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    height, width = x.shape[-2:]
    # Four pooling stages require dimensions divisible by 16. Crop after prediction.
    x = F.pad(x[None], (0, (-width) % 16, 0, (-height) % 16), mode="replicate")
    logits = model(x.to(device))
    mask = logits.argmax(dim=1)[0, :height, :width].cpu().numpy().astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    print(f"Saved {width} x {height} class-ID mask to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    predict(args.checkpoint, args.image, args.output, args.device, args.num_classes, args.base)


if __name__ == "__main__":
    main()
