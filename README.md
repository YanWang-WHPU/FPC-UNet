# FPC-UNet

**FPC-UNet: A Dual-Stage Calibration Network for UAV-Based Rice Field Segmentation**

Kaijun Yang, Yan Wang, Guangwei Liu, Sheng Wang, Xiaohui Huang

[Paper (Springer)](https://link.springer.com/chapter/10.1007/978-981-92-6621-0_3) · [DOI](https://doi.org/10.1007/978-981-92-6621-0_3) · [BibTeX](citation.bib) · [中文说明](README_zh-CN.md)

Author-provided PyTorch implementation accompanying the published paper in *Web and Big Data. APWeb-WAIM 2026 International Workshops*, CCIS 3188, pp. 27–38.

**This is a code-only release. The research dataset, UAV images, annotations, dataset splits, trained weights and experiment outputs are not distributed.** Training and inference require your own data and locally trained checkpoints. No sample images from the research dataset are included.

## Overview

FPC-UNet retains a U-Net encoder–decoder and calibrates its predictions in two stages:

- **Full-Scale Context Calibration (FSCC)** projects features from all four encoder stages and the bottleneck, learns scale-wise fusion weights, and adds a gated residual context term to the final decoder feature.
- **Prototype-Guided Logit Calibration (PGLC)** uses normalized feature embeddings and learnable class prototypes to produce bounded, class-wise corrections to the segmentation logits.

```text
RGB image → U-Net encoder / decoder → FSCC → prediction head → PGLC → class logits
                 │                     ↑
                 └── encoder and bottleneck features
```

The default model uses 3 input channels, 4 output classes and a base width of 32. It returns raw logits with shape `[B, num_classes, H, W]`. For direct model calls, both spatial dimensions must be divisible by 16; the paper uses 512 × 512 patches. The inference utility pads other sizes and crops the output back to its original dimensions.

## Repository layout

```text
extracted_fpc_unet/                 Standalone FPC-UNet; depends only on PyTorch
  fpc_unet.py                       U-Net backbone, FSCC and PGLC
experiment/
  models.py                         Original experiment model factory and ablations
  train_segmentation.py             Training, validation, checkpointing and test evaluation
ourModel/bfca_modules.py             Calibration and supporting experiment modules
scripts/
  predict.py                        Predict a class-ID PNG from one RGB image
  evaluate_segmentation_outputs.py  Evaluate saved class-ID masks
tests/                              Synthetic model and command-line workflow checks
requirements-model.txt              Standalone model dependency
requirements.txt                    Training / evaluation dependencies
CITATION.cff                        GitHub citation metadata
citation.bib                        Paper reference
```

The standalone implementation and the experiment factory's `build_model("fpc_unet")` have matching state-dictionary keys. Existing FPC-UNet checkpoints from the original project can be loaded by either implementation. The experiment name `unet_fscc_pglc` is an alias of `fpc_unet`. Other model names in `experiment/models.py` are the project's experiment implementations; they are not distributions of the corresponding methods' official repositories.

## Installation

Use Python 3.10 or newer. Create and activate a virtual environment, then install the dependencies from the repository root:

```bash
git clone https://github.com/YanWang-WHPU/FPC-UNet.git
cd FPC-UNet
python -m pip install -r requirements.txt
```

For CUDA, install a matching PyTorch / torchvision pair for your system using the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Training automatically selects CUDA when available and otherwise runs on CPU.

To use only the standalone model:

```bash
python -m pip install -r requirements-model.txt
```

Release checks were run with Python 3.10, PyTorch 2.8.0, torchvision 0.23.0, timm 1.0.27 and NumPy 2.2.6. These are the release-validation versions, not a claim about the original training environment.

## Model quick start

This example uses synthetic input and random initialization; it needs no dataset or checkpoint.

```python
import torch
from extracted_fpc_unet import build_fpc_unet

model = build_fpc_unet(num_classes=4, base=32, in_channels=3).eval()
images = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    logits = model(images)
    labels = logits.argmax(dim=1)
print(logits.shape)  # torch.Size([1, 4, 64, 64])
print(labels.shape)  # torch.Size([1, 64, 64])
```

For training or inference on real RGB images, scale pixels to `[0, 1]` and normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. The provided loader and inference utility already do this.

## Prepare your own data

The original dataset and split manifests are not included. Prepare RGB image tiles and aligned, single-channel integer label masks. Masks must contain class IDs, not RGB visualization colors. All samples within a batch must have the same height and width, divisible by 16. Use square patches for the training loader's 90-degree rotation augmentation.

The supplied training entry point is configured for these four classes:

| ID | Class | Visualization color |
| --- | --- | --- |
| 0 | Background | Black `(0, 0, 0)` |
| 1 | Mid-season Rice | Green `(0, 255, 0)` |
| 2 | Late Rice | Yellow `(255, 255, 0)` |
| 3 | Lotus | Red `(255, 0, 0)` |

No ignore label is configured in the training loss; masks must contain only IDs 0–3. The standalone model accepts a different `num_classes`; adapting the training script also requires updating its model construction, class names, palette and foreground metric indices.

Example **local** layout (none of these data files are shipped):

```text
data/
  images/patch_001.png
  masks/patch_001.png
  splits/train.csv
  splits/val.csv
  splits/test.csv
```

Each CSV must have `image` and `mask` columns. The following is an illustrative row, not a research split:

```csv
image,mask
data/images/patch_001.png,data/masks/patch_001.png
```

Paths may be absolute or relative to the repository root when commands are run there. Use distinct samples for training, validation and testing, and keep overlapping spatial regions in the same split to avoid leakage. The loader does not reconstruct the paper's private splits. Geospatial alignment, orthomosaic cropping and annotation rasterization must be completed with your own tools before training; ArcGIS and the original site's preprocessing workflows are not required by this release.

## Training

Run all commands below from the repository root. This command uses the paper's optimization settings on your own 512 × 512 tiles:

```bash
python -m experiment.train_segmentation --model fpc_unet --splits-dir data/splits --out-root outputs/experiments --epochs 80 --batch-size 8 --lr 0.0003 --weight-decay 0.0001 --optimizer adamw --scheduler cosine --num-workers 4
```

Use a smaller batch if GPU memory is limited; at least one full training batch is needed because the training loader uses `drop_last=True`. For CPU execution, add `--device cpu --skip-profile` to select CPU and avoid the 512 × 512 latency benchmark. On Windows, `--num-workers 0` can simplify data-loader setup. `--amp` enables mixed precision on CUDA; it is optional and off by default. Augmentation includes random flips, 90-degree rotations and brightness / contrast / saturation adjustments. The loss is cross-entropy, and the best checkpoint is selected by validation mIoU over all four classes. A fixed seed is provided, but bitwise deterministic training is not guaranteed.

Outputs are written beneath `outputs/experiments/fpc_unet/`, including `checkpoints/best.pt`, `checkpoints/last.pt`, training curves, logs, prediction visualizations and validation / test metrics. Generated outputs remain local and are ignored by Git.

Resume an interrupted run with the same settings and `--resume`:

```bash
python -m experiment.train_segmentation --model fpc_unet --splits-dir data/splits --out-root outputs/experiments --epochs 80 --batch-size 8 --resume
```

The existing experiment factory also provides the core ablations:

| Model name | FSCC | PGLC |
| --- | --- | --- |
| `unet` | No | No |
| `unet_fscc` | Yes | No |
| `unet_pglc` | No | Yes |
| `fpc_unet` / `unet_fscc_pglc` | Yes | Yes |

The paper's results depend on its private dataset, spatial splits and training runs. Running these settings on different data does not reproduce those numerical results.

## Evaluation

Evaluate your own checkpoint using your own manifests:

```bash
python -m experiment.train_segmentation --model fpc_unet --splits-dir data/splits --out-root outputs/evaluation --eval-only --eval-checkpoint outputs/experiments/fpc_unet/checkpoints/best.pt --skip-profile
```

The script writes overall accuracy (OA), mean intersection over union (mIoU), mean F1, class-wise metrics and a confusion matrix. Reports include all-class means and foreground-only means. In the `ignore_background` report, background is excluded from the mean over classes; background pixels still contribute to the confusion matrix and OA. Absent classes contribute zero to the corresponding mean in this implementation. See `scripts/evaluate_segmentation_outputs.py` for the exact definitions.

To evaluate separately saved masks, prediction and ground-truth filenames must match. Both must be single-channel class-ID PNGs:

```bash
python -m scripts.evaluate_segmentation_outputs --pred-dir outputs/predictions --gt-dir data/masks --out-dir outputs/metrics
```

## Single-image inference

Train a model first, or supply a compatible checkpoint you already own. No pretrained checkpoint is downloaded automatically.

```bash
python -m scripts.predict --checkpoint outputs/experiments/fpc_unet/checkpoints/best.pt --image data/images/patch_001.png --output outputs/predictions/patch_001.png
```

Add `--device cpu` to select CPU inference. The script accepts the training checkpoint dictionary containing `model` or a plain state dictionary. It saves the original image size as an 8-bit PNG containing class IDs 0–3, so the mask can appear nearly black in an ordinary image viewer. `--num-classes` and `--base` must match your checkpoint. This utility processes one image at a time; tile very large UAV orthomosaics before use.

## Synthetic checks

After installing the full dependencies, run:

```bash
python -m unittest discover -s tests -v
```

Checks verify exact output equivalence between the standalone model and original experiment implementation with active calibration, gradient flow through both calibration branches, and a one-epoch CPU training / checkpoint / inference / evaluation workflow. They use random tensors and temporary synthetic files, with no research data. The workflow check briefly creates local checkpoints and deletes them on completion.

## Citation

Please cite the associated paper when using this implementation. Springer lists **2027** as the citation year; the chapter was first published online on **15 October 2026**.

```bibtex
@inproceedings{yang2027fpcunet,
  author    = {Yang, Kaijun and Wang, Yan and Liu, Guangwei and Wang, Sheng and Huang, Xiaohui},
  title     = {{FPC-UNet}: A Dual-Stage Calibration Network for {UAV}-Based Rice Field Segmentation},
  booktitle = {Web and Big Data. APWeb-WAIM 2026 International Workshops},
  series    = {Communications in Computer and Information Science},
  volume    = {3188},
  pages     = {27--38},
  year      = {2027},
  publisher = {Springer, Singapore},
  doi       = {10.1007/978-981-92-6621-0_3}
}
```

For questions about the code, please open an [issue](https://github.com/YanWang-WHPU/FPC-UNet/issues).
