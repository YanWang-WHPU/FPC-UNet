# FPC-UNet

论文 **FPC-UNet: A Dual-Stage Calibration Network for UAV-Based Rice Field Segmentation** 的 PyTorch 代码。

作者：Kaijun Yang、Yan Wang、Guangwei Liu、Sheng Wang、Xiaohui Huang。

[论文页面](https://link.springer.com/chapter/10.1007/978-981-92-6621-0_3) · [英文完整说明](README.md) · [BibTeX](citation.bib)

**本仓库仅公开代码，不提供研究数据集、无人机影像、标注、数据划分清单、训练权重或实验结果文件，也不包含原始数据的样例图像。** 请使用自己的数据和本地训练的模型。

## 方法与代码

FPC-UNet 在 U-Net 上引入两个校准模块：FSCC 汇聚编码器各层和瓶颈层的特征，对最终解码特征进行残差校准；PGLC 利用可学习的类别原型，对分割 logits 进行有界的类别校正。

- `extracted_fpc_unet/`：独立模型，仅依赖 PyTorch，默认 RGB 输入、4 类输出、基础通道数 32。
- `experiment/`：原实验模型工厂、训练、验证与测试入口，保留核心消融模型。
- `ourModel/`：原实验使用的校准及辅助模块。
- `scripts/predict.py`：使用自己的权重进行单张 RGB 图像推理。
- `scripts/evaluate_segmentation_outputs.py`：计算标签图的 OA、mIoU、F1 与混淆矩阵。
- `tests/`：使用随机张量检查独立模型与原实现的一致性，以及校准模块的梯度。

公开版本保留原模型计算和权重键名，整理了文件范围和默认路径，将训练默认 batch size 调整为论文中的 8，增加通用推理入口、设备选择和 `--skip-profile` 选项。原始研究文件夹中的场景专用预处理、绘图、文稿和自动化脚本不在发布范围内。

## 安装和快速使用

建议使用 Python 3.10 或更高版本，并在虚拟环境中安装依赖。CUDA 用户需按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配的 PyTorch 与 torchvision。

```bash
git clone https://github.com/YanWang-WHPU/FPC-UNet.git
cd FPC-UNet
python -m pip install -r requirements.txt
```

如果只使用独立模型，安装 `requirements-model.txt` 即可。下面的例子不需要数据或权重：

```python
import torch
from extracted_fpc_unet import build_fpc_unet

model = build_fpc_unet(num_classes=4).eval()
with torch.no_grad():
    logits = model(torch.randn(1, 3, 64, 64))
print(logits.shape)  # [1, 4, 64, 64]
```

直接调用模型时，输入宽高均需为 16 的倍数。真实图像需缩放至 `[0, 1]`，并采用 ImageNet 均值和标准差归一化；训练和推理脚本已实现此步骤。

## 自备数据

训练代码采用四类标签：`0=背景`、`1=中稻`、`2=晚稻`、`3=莲藕`。标注必须是与 RGB 影像对齐的单通道类别 ID 图，不是彩色可视化图；当前训练损失不支持额外的忽略标签。建议准备 512 × 512 的方形图块。

在 `data/splits/` 下自建 `train.csv`、`val.csv` 和 `test.csv`，每个文件包含 `image,mask` 两列。例如：

```csv
image,mask
data/images/patch_001.png,data/masks/patch_001.png
```

这些仅是路径示例，仓库不附带任何图像或真实清单。路径可使用绝对路径，或相对于运行命令时仓库根目录的路径。训练、验证、测试应保持空间独立，避免重叠区域跨集合造成数据泄漏。

## 训练、评估和推理

以下命令均在仓库根目录运行。论文使用 80 个 epoch、batch size 8、AdamW、初始学习率 0.0003、权重衰减 0.0001 和余弦退火：

```bash
python -m experiment.train_segmentation --model fpc_unet --splits-dir data/splits --out-root outputs/experiments --epochs 80 --batch-size 8 --lr 0.0003 --weight-decay 0.0001 --optimizer adamw --scheduler cosine
```

Windows 下可使用 `--num-workers 0`；CPU 下可增加 `--device cpu --skip-profile` 指定设备并跳过耗时的性能测试。训练样本数至少需要达到一个完整 batch。最优权重根据验证集四类平均 mIoU 选择，输出保存在 `outputs/experiments/fpc_unet/`，不会纳入 Git。

评估自己的权重：

```bash
python -m experiment.train_segmentation --model fpc_unet --splits-dir data/splits --out-root outputs/evaluation --eval-only --eval-checkpoint outputs/experiments/fpc_unet/checkpoints/best.pt --skip-profile
```

单图推理：

```bash
python -m scripts.predict --checkpoint outputs/experiments/fpc_unet/checkpoints/best.pt --image data/images/patch_001.png --output outputs/predictions/patch_001.png
```

推理输出为原图尺寸的类别 ID PNG，数值为 0–3，在普通看图软件中可能接近全黑。非 16 倍数的输入会先补齐，再将输出裁回原尺寸。超大正射影像请先分块。

运行不依赖研究数据的检查：

```bash
python -m unittest discover -s tests -v
```

详细的数据格式、消融模型、指标定义和命令参数见[英文 README](README.md)。本仓库不提供私有数据上的数值复现材料，使用不同数据集运行上述设置不能保证得到论文中的结果。

## 引用

请使用 [citation.bib](citation.bib) 或 GitHub 的 “Cite this repository”。Springer 推荐引用年份为 **2027**，在线发表日期为 **2026 年 10 月 15 日**，两者按出版页面分别记录。
