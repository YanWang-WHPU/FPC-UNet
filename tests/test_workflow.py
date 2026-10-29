"""Exercise public CLI commands end to end using temporary synthetic data."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
        result = subprocess.run(
            [sys.executable, "-m", *map(str, args)], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_train_checkpoint_predict_and_evaluate(self):
        with tempfile.TemporaryDirectory(prefix="fpc-unet-test-") as directory:
            work = Path(directory)
            rng = np.random.default_rng(17)
            splits = work / "splits"
            splits.mkdir()
            for split in ("train", "val", "test"):
                with (splits / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["image", "mask"])
                    writer.writeheader()
                    for index in range(2):
                        image = work / f"{split}_{index}.png"
                        mask = work / f"{split}_{index}_mask.png"
                        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(image)
                        Image.fromarray(rng.integers(0, 4, (32, 32), dtype=np.uint8)).save(mask)
                        writer.writerow({"image": str(image), "mask": str(mask)})
            self.run_cli(
                "experiment.train_segmentation", "--model", "fpc_unet",
                "--splits-dir", splits, "--out-root", work / "runs",
                "--epochs", "1", "--batch-size", "2", "--num-workers", "0",
                "--device", "cpu",
                "--vis-count", "0", "--limit-batches", "1", "--skip-profile",
            )
            run = work / "runs" / "fpc_unet"
            checkpoint = run / "checkpoints" / "best.pt"
            self.assertTrue(checkpoint.is_file())
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["best_epoch"], 1)
            self.assertTrue(np.isfinite(summary["test_loss"]))
            image = work / "odd_size.png"
            Image.fromarray(rng.integers(0, 256, (33, 47, 3), dtype=np.uint8)).save(image)
            prediction = work / "predictions" / "odd_size.png"
            self.run_cli(
                "scripts.predict", "--checkpoint", checkpoint,
                "--image", image, "--output", prediction, "--device", "cpu",
            )
            with Image.open(prediction) as result:
                labels = np.array(result)
            self.assertEqual(labels.shape, (33, 47))
            self.assertTrue(np.isin(labels, [0, 1, 2, 3]).all())
            gt_dir = work / "ground_truth"
            gt_dir.mkdir()
            Image.fromarray(rng.integers(0, 4, (33, 47), dtype=np.uint8)).save(gt_dir / prediction.name)
            self.run_cli(
                "scripts.evaluate_segmentation_outputs", "--pred-dir", prediction.parent,
                "--gt-dir", gt_dir, "--out-dir", work / "metrics",
            )
            metrics = json.loads((work / "metrics" / "overall_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["num_images"], 1)
            self.assertEqual(metrics["with_background"]["pixels"], 33 * 47)


if __name__ == "__main__":
    unittest.main()
