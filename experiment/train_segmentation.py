from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from thop import profile
from torch import nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from experiment.models import build_model
from scripts.evaluate_segmentation_outputs import fast_confusion_matrix, metric_dict, write_confusion, write_csv


CLASS_NAMES = {
    0: "Background",
    1: "Mid-season Rice",
    2: "Late Rice",
    3: "Lotus",
}

PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 255, 0],
        [255, 255, 0],
        [255, 0, 0],
    ],
    dtype=np.uint8,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class RiceDataset(Dataset):
    def __init__(self, split_csv: Path, train: bool = False):
        self.rows = []
        with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        mask = Image.open(row["mask"])
        if self.train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            k = random.randint(0, 3)
            if k:
                image = image.rotate(90 * k, expand=False)
                mask = mask.rotate(90 * k, expand=False)
            if random.random() < 0.35:
                image = TF.adjust_brightness(image, random.uniform(0.85, 1.15))
                image = TF.adjust_contrast(image, random.uniform(0.85, 1.15))
                image = TF.adjust_saturation(image, random.uniform(0.85, 1.15))
        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        y = torch.from_numpy(np.array(mask, dtype=np.int64))
        return x, y, Path(row["image"]).stem


class ConfusionMeter:
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        target_np = target.detach().cpu().numpy().astype(np.int64)
        self.confusion += fast_confusion_matrix(target_np, pred_np, self.num_classes, ignore_index=None)

    def metrics(self) -> dict[str, float]:
        overall, _ = metric_dict(self.confusion, CLASS_NAMES, range(self.num_classes))
        return {"OA": overall["OA"], "mIoU": overall["mIoU"], "mF1": overall["mF1"]}


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    arr = (x * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    return (arr * 255).astype(np.uint8)


def boundary_target(mask: torch.Tensor, dilation: int = 3) -> torch.Tensor:
    target = torch.zeros_like(mask, dtype=torch.float32)
    target[:, :, 1:] = torch.maximum(target[:, :, 1:], (mask[:, :, 1:] != mask[:, :, :-1]).float())
    target[:, :, :-1] = torch.maximum(target[:, :, :-1], (mask[:, :, 1:] != mask[:, :, :-1]).float())
    target[:, 1:, :] = torch.maximum(target[:, 1:, :], (mask[:, 1:, :] != mask[:, :-1, :]).float())
    target[:, :-1, :] = torch.maximum(target[:, :-1, :], (mask[:, 1:, :] != mask[:, :-1, :]).float())
    if dilation > 1:
        target = F.max_pool2d(target[:, None], kernel_size=dilation, stride=1, padding=dilation // 2)[:, 0]
    return target[:, None]


def forward_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module, boundary_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    if boundary_weight <= 0:
        logits = model(x)
        return criterion(logits, y), logits
    try:
        logits, aux = model(x, return_aux=True)
    except TypeError:
        logits = model(x)
        return criterion(logits, y), logits
    loss = criterion(logits, y)
    boundaries = aux.get("boundaries", []) if isinstance(aux, dict) else []
    if boundaries:
        target = boundary_target(y)
        boundary_loss = torch.zeros((), device=x.device)
        with autocast("cuda", enabled=False):
            for boundary in boundaries:
                scaled = F.interpolate(target, size=boundary.shape[-2:], mode="nearest")
                boundary_loss = boundary_loss + F.binary_cross_entropy(boundary.float(), scaled.float())
        loss = loss + boundary_weight * (boundary_loss / len(boundaries))
    return loss, logits


def save_prediction_triplet(image: torch.Tensor, pred: np.ndarray, target: np.ndarray, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = denormalize(image)
    pred_color = colorize(pred)
    gt_color = colorize(target)
    overlay = (0.55 * rgb + 0.45 * pred_color).astype(np.uint8)
    error = rgb.copy()
    error[pred != target] = [255, 0, 0]
    Image.fromarray(pred.astype(np.uint8)).save(out_dir / f"{stem}_pred.png")
    Image.fromarray(pred_color).save(out_dir / f"{stem}_pred_color.png")
    Image.fromarray(gt_color).save(out_dir / f"{stem}_gt_color.png")
    Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")
    Image.fromarray(error).save(out_dir / f"{stem}_error.png")


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, limit_batches: int = 0, boundary_weight: float = 0.0) -> tuple[float, dict[str, float]]:
    model.train()
    meter = ConfusionMeter()
    total_loss = 0.0
    total = 0
    for batch_idx, (x, y, _) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=scaler is not None):
            loss, logits = forward_loss(model, x, y, criterion, boundary_weight)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        pred = logits.argmax(dim=1)
        meter.update(pred, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if limit_batches and batch_idx >= limit_batches:
            break
    return total_loss / max(total, 1), meter.metrics()


@torch.no_grad()
def evaluate(model, loader, criterion, device, pred_dir: Path | None = None, max_vis: int = 80, limit_batches: int = 0) -> tuple[float, dict[str, float], np.ndarray]:
    model.eval()
    meter = ConfusionMeter()
    total_loss = 0.0
    total = 0
    saved = 0
    for batch_idx, (x, y, stems) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        meter.update(pred, y)
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        if pred_dir is not None and saved < max_vis:
            for i, stem in enumerate(stems):
                if saved >= max_vis:
                    break
                save_prediction_triplet(x[i], pred[i].cpu().numpy(), y[i].cpu().numpy(), pred_dir, stem)
                saved += 1
        if limit_batches and batch_idx >= limit_batches:
            break
    return total_loss / max(total, 1), meter.metrics(), meter.confusion


def save_metrics(confusion: np.ndarray, out_dir: Path, speed: dict | None = None) -> dict:
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    overall_all, per_class = metric_dict(confusion, CLASS_NAMES, range(confusion.shape[0]))
    overall_fg, _ = metric_dict(confusion, CLASS_NAMES, [1, 2, 3])
    payload = {
        "with_background": overall_all,
        "ignore_background": overall_fg,
    }
    if speed:
        payload["speed"] = speed
    (metrics_dir / "overall_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(metrics_dir / "overall_metrics.csv", [{"scope": "with_background", **overall_all}, {"scope": "ignore_background", **overall_fg}])
    write_csv(metrics_dir / "per_class_metrics.csv", per_class)
    write_confusion(metrics_dir / "confusion_matrix.csv", confusion, CLASS_NAMES)
    return payload


def plot_curves(log_rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in log_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=180)
    for ax, title, keys in [
        (axes[0], "Loss", ["train_loss", "val_loss"]),
        (axes[1], "OA", ["train_OA", "val_OA"]),
        (axes[2], "mIoU", ["train_mIoU", "val_mIoU"]),
    ]:
        for key in keys:
            ax.plot(epochs, [row[key] for row in log_rows], label=key)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_log(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile_model(model: nn.Module, device: torch.device, batch_size: int = 1) -> dict:
    model.eval()
    dummy = torch.randn(batch_size, 3, 512, 512, device=device)
    params = sum(p.numel() for p in model.parameters())
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        macs = 0
    warmup = 10
    repeats = 50
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    latency = elapsed / repeats / batch_size
    return {
        "params": int(params),
        "params_m": params / 1e6,
        "macs": int(macs),
        "macs_g": macs / 1e9,
        "latency_ms_per_image": latency * 1000.0,
        "fps": 1.0 / latency if latency > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a specific device such as cuda:0.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "poly", "none"], default="cosine")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--out-root", default="outputs/experiments")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate the best checkpoint in the output directory.")
    parser.add_argument("--eval-checkpoint", default="", help="Checkpoint path to evaluate instead of the best checkpoint.")
    parser.add_argument("--init-checkpoint", default="", help="Initialize matching model weights from a checkpoint before training.")
    parser.add_argument("--freeze-loaded", action="store_true", help="Freeze parameters loaded from --init-checkpoint and train only new modules.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--vis-count", type=int, default=120)
    parser.add_argument("--limit-batches", type=int, default=0, help="Debug only: limit train/val/test batches per epoch.")
    parser.add_argument("--save-every", type=int, default=0, help="Save an epoch checkpoint every N epochs.")
    parser.add_argument("--skip-profile", action="store_true", help="Skip the 512x512 MACs and latency benchmark (useful on CPU).")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    out_dir = Path(args.out_root) / args.model
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RiceDataset(Path(args.splits_dir) / "train.csv", train=True)
    val_ds = RiceDataset(Path(args.splits_dir) / "val.csv", train=False)
    test_ds = RiceDataset(Path(args.splits_dir) / "test.csv", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes=4).to(device)
    if args.init_checkpoint and not args.resume:
        init_state = torch.load(args.init_checkpoint, map_location=device)
        state_dict = init_state.get("model", init_state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if args.freeze_loaded:
            loaded_keys = set(state_dict.keys()) - set(unexpected)
            for name, param in model.named_parameters():
                if name in loaded_keys:
                    param.requires_grad = False
        print(
            f"Initialized from {args.init_checkpoint}: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    criterion = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "poly":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max(args.epochs, 1)) ** 0.9)
    elif args.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler("cuda") if args.amp and device.type == "cuda" else None

    start_epoch = 1
    best_miou = -1.0
    log_rows: list[dict] = []
    last_ckpt = ckpt_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_miou = state["best_miou"]
        start_epoch = state["epoch"] + 1
        log_rows = state.get("log_rows", [])

    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, args.limit_batches, args.boundary_loss_weight)
            val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device, limit_batches=args.limit_batches)
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_OA": train_metrics["OA"],
                "train_mIoU": train_metrics["mIoU"],
                "train_mF1": train_metrics["mF1"],
                "val_OA": val_metrics["OA"],
                "val_mIoU": val_metrics["mIoU"],
                "val_mF1": val_metrics["mF1"],
            }
            log_rows.append(row)
            write_log(out_dir / "logs" / "train_log.csv", log_rows)
            plot_curves(log_rows, out_dir / "curves" / "training_curves.png")
            is_best = val_metrics["mIoU"] > best_miou
            if is_best:
                best_miou = val_metrics["mIoU"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_miou": best_miou, "args": vars(args)}, ckpt_dir / "best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "log_rows": log_rows,
                    "args": vars(args),
                },
                last_ckpt,
            )
            if args.save_every and epoch % args.save_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_miou": best_miou,
                        "log_rows": log_rows,
                        "args": vars(args),
                    },
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                )
            print(
                f"[{args.model}] epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_OA={val_metrics['OA']:.4f} val_mIoU={val_metrics['mIoU']:.4f} val_mF1={val_metrics['mF1']:.4f} "
                f"best_mIoU={best_miou:.4f}",
                flush=True,
            )

    best_path = Path(args.eval_checkpoint) if args.eval_checkpoint else ckpt_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint for evaluation: {best_path}")
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])
    speed = {} if args.skip_profile else profile_model(model, device)
    (out_dir / "speed").mkdir(parents=True, exist_ok=True)
    (out_dir / "speed" / "model_profile.json").write_text(json.dumps(speed, indent=2), encoding="utf-8")

    val_loss, val_metrics, val_conf = evaluate(model, val_loader, criterion, device, out_dir / "predictions" / "val", args.vis_count, args.limit_batches)
    test_loss, test_metrics, test_conf = evaluate(model, test_loader, criterion, device, out_dir / "predictions" / "test", args.vis_count, args.limit_batches)
    val_payload = save_metrics(val_conf, out_dir / "val_results", speed)
    test_payload = save_metrics(test_conf, out_dir / "test_results", speed)
    summary = {
        "model": args.model,
        "best_epoch": int(best["epoch"]),
        "splits_dir": args.splits_dir,
        "boundary_loss_weight": args.boundary_loss_weight,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val": val_payload,
        "test": test_payload,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
