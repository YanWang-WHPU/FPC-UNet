"""Evaluate semantic segmentation predictions and save paper-ready metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_CLASSES = {
    0: "Background",
    1: "Mid-season Rice",
    2: "Late Rice",
    3: "Lotus",
}


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def fast_confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int | None) -> np.ndarray:
    valid = (gt >= 0) & (gt < num_classes)
    if ignore_index is not None:
        valid &= gt != ignore_index
    encoded = num_classes * gt[valid] + pred[valid]
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metric_dict(confusion: np.ndarray, class_names: dict[int, str], classes_for_mean: Iterable[int]) -> tuple[dict, list[dict]]:
    eps = 1e-12
    tp = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp

    iou = tp / np.maximum(union, eps)
    class_oa = tp / np.maximum(gt_count, eps)
    precision = tp / np.maximum(pred_count, eps)
    f1 = 2.0 * precision * class_oa / np.maximum(precision + class_oa, eps)

    classes = list(classes_for_mean)
    overall = {
        "OA": float(tp.sum() / max(confusion.sum(), eps)),
        "mIoU": float(np.mean(iou[classes])) if classes else 0.0,
        "mF1": float(np.mean(f1[classes])) if classes else 0.0,
        "pixels": int(confusion.sum()),
    }
    per_class = []
    for idx in range(confusion.shape[0]):
        per_class.append(
            {
                "class_id": idx,
                "class_name": class_names.get(idx, str(idx)),
                "IoU": float(iou[idx]),
                "OA": float(class_oa[idx]),
                "F1": float(f1[idx]),
                "precision": float(precision[idx]),
                "gt_pixels": int(gt_count[idx]),
                "pred_pixels": int(pred_count[idx]),
                "tp_pixels": int(tp[idx]),
            }
        )
    return overall, per_class


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(path: Path, confusion: np.ndarray, class_names: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["gt\\pred"] + [class_names.get(i, str(i)) for i in range(confusion.shape[0])]
        writer.writerow(header)
        for idx, row in enumerate(confusion):
            writer.writerow([class_names.get(idx, str(idx)), *row.tolist()])


def load_class_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return DEFAULT_CLASSES
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data.get("classes", data)
    return {int(k): str(v) for k, v in classes.items()}


def prediction_pairs(pred_dir: Path, gt_dir: Path, suffix: str) -> list[tuple[Path, Path]]:
    pairs = []
    for pred in sorted(pred_dir.glob(f"*{suffix}")):
        gt = gt_dir / pred.name
        if gt.exists():
            pairs.append((pred, gt))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted label masks.")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground-truth label masks.")
    parser.add_argument("--out-dir", required=True, help="Output directory for metrics.")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--metadata", default=None, help="Optional JSON class names; otherwise use the four paper classes.")
    parser.add_argument("--suffix", default=".png")
    parser.add_argument("--ignore-index", type=int, default=None)
    parser.add_argument("--background-class", type=int, default=0)
    parser.add_argument("--speed-json", default=None, help="Optional speed/model profile JSON to merge into overall metrics.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    class_names = load_class_names(Path(args.metadata) if args.metadata else None)

    pairs = prediction_pairs(pred_dir, gt_dir, args.suffix)
    if not pairs:
        raise SystemExit(f"No prediction/ground-truth pairs found in {pred_dir} and {gt_dir}.")

    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for pred_path, gt_path in pairs:
        pred = load_mask(pred_path)
        gt = load_mask(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch: {pred_path} {pred.shape} vs {gt_path} {gt.shape}")
        confusion += fast_confusion_matrix(gt, pred, args.num_classes, args.ignore_index)

    all_classes = list(range(args.num_classes))
    fg_classes = [idx for idx in all_classes if idx != args.background_class]

    overall_all, per_class = metric_dict(confusion, class_names, all_classes)
    overall_fg, _ = metric_dict(confusion, class_names, fg_classes)
    overall = {
        "num_images": len(pairs),
        "with_background": overall_all,
        "ignore_background": overall_fg,
    }

    if args.speed_json:
        speed_path = Path(args.speed_json)
        if speed_path.exists():
            overall["speed"] = json.loads(speed_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "overall_metrics.csv",
        [
            {"scope": "with_background", **overall_all},
            {"scope": "ignore_background", **overall_fg},
        ],
    )
    write_csv(out_dir / "per_class_metrics.csv", per_class)
    write_confusion(out_dir / "confusion_matrix.csv", confusion, class_names)

    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
