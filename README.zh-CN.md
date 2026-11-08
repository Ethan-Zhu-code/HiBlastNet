<p align="center"><a href="README.md">English</a> &nbsp;·&nbsp; <strong>简体中文</strong></p>

<p align="center">
  <img src=".github/media/hiblastnet-banner-clean.png" alt="HiBlastNet 水稻叶瘟病斑分割示意图" width="100%" />
</p>

<p align="center"><strong>在标定后的整叶图像上分割水稻叶瘟病斑。</strong></p>

<p align="center">
  <a href="#安装">安装</a> &nbsp;·&nbsp;
  <a href="#数据集">数据集</a> &nbsp;·&nbsp;
  <a href="#预训练权重">权重</a> &nbsp;·&nbsp;
  <a href="#训练">训练</a> &nbsp;·&nbsp;
  <a href="#评估">评估</a> &nbsp;·&nbsp;
  <a href="#单叶预测">单叶预测</a> &nbsp;·&nbsp;
  <a href="#原图评级">原图评级</a> &nbsp;·&nbsp;
  <a href="#方法与配置">方法</a> &nbsp;·&nbsp;
  <a href="#引用">引用</a>
</p>

---

本仓库提供论文 *Preserving small lesions in full-leaf segmentation of rice blast: a calibrated workflow for lesion morphology and severity measurement* 的分割模型训练与评估代码。HiBlastNet 将图像中的每个像素分为背景、健康叶组织、感病型病斑或抗病型病斑。

| 保留叶片细节 | 细化病斑边界 | 监督病斑组件 |
| --- | --- | --- |
| 在标定分辨率下，以相互重叠的 1024 × 640 窗口推理，再用 Hann 权重融合各窗口的 logits。 | BC-MSRR 细化分支结合局部、上下文和边界特征。 | LCJS 在训练时加入尺寸感知、组件局部和健康组织监督，推理时不增加计算。 |

仓库还提供从标定板原图到叶片提取、分割和 0–9 级评级的命令行流程。桌面应用单独维护。

## 安装

安装 Git 和 Miniconda 后，执行：

```bash
git clone https://github.com/Ethan-Zhu-code/HiBlastNet.git
cd HiBlastNet
conda create -n hiblastnet python=3.11 -y
conda activate hiblastnet
```

根据硬件选择 [PyTorch 官网](https://pytorch.org/get-started/locally/) 提供的命令安装 PyTorch 和 torchvision，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

论文实验使用 PyTorch 2.5.1 和 CUDA 12.1。训练需要 NVIDIA GPU；评估和预测也可在 CPU 上运行，但速度较慢。以下命令均在 `HiBlastNet` 项目目录执行。

## 数据集

**BlastLeaf** 包含从 283 张离体水稻叶片照片中提取的 707 张单叶感兴趣区域（ROI）及像素级标注。任选一个网盘下载：

| 网盘 | 链接 | 提取码 |
| --- | --- | --- |
| OneDrive | [下载](https://1drv.ms/u/c/1495c176035c0989/IQC3HxyG_ktHSpHlq_E-5vmXAYjrLFhmseHouaPyHpjrC-4?e=SkNipz) | — |
| 百度网盘 | [下载](https://pan.baidu.com/s/1K6AEge89TWjPSs5MAovVrg?pwd=p1p6) | `p1p6` |

数据集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可，注明来源后可分享和改编，包括商业用途。解压后整理为以下目录结构：

```text
HiBlastNet/
  data/blastleaf/
    manifest.csv
    split_summary.json
    train/   images/  masks/  relations/
    val/     images/  masks/  relations/
    test/    images/  masks/  relations/
```

**图像。** 每张图像对应一片叶子。原始照片经标定板校正至 60 像素/mm 后裁切，叶片长轴沿竖直方向。

**掩膜。** 单通道 PNG，类别定义如下：

| 标签 | 含义 |
| --- | --- |
| 0 | 背景 |
| 1 | 健康叶组织，也包含非稻瘟病造成的损伤 |
| 2 | 感病型病斑 |
| 3 | 抗病型病斑 |

病斑类型描述的是形态，不等同于植株的抗性等级。

**划分。** 训练集、验证集和测试集分别含 503、104 和 100 片叶子。同一原始照片中的叶片始终放在同一集合。比较实验结果时请沿用此划分。

**清单。** `manifest.csv` 每行记录一片叶子的 `sample_id`、`split`、`source_image_id`、`leaf_id`、图像/掩膜/关系文件的相对路径、宽高和四类像素数。`split_summary.json` 记录原图与数据划分的对应关系，`relations/*.json` 保存逐叶标识、尺寸和路径。数据加载器会检查这些文件的一致性。使用自有数据训练时请沿用该结构，详见 [数据加载代码](blastseg/data/full_leaf.py)。

### 原图示例

**`BlastLeaf_30_Original_Photos.zip`**（419 MB）包含 BlastLeaf 测试集的 30 张标定板原始 JPEG，对应 72 片已标注的叶片。该压缩包与裁切后的 ROI 数据集分开下载，内附原图编号、原始文件名和 SHA-256 校验值。

| 网盘 | 链接 | 提取码 |
| --- | --- | --- |
| OneDrive | [下载](https://1drv.ms/u/c/1495c176035c0989/IQBtNABm1JC3SIWQSa1nlXPRAYTeY_4OhdFh1d5rHNjfLeA?e=g9yzBJ) | — |
| 百度网盘 | [下载](https://pan.baidu.com/s/1EUgH1kOioeeU1JFz02Q_cw?pwd=ek4a) | `ek4a` |

## 预训练权重

论文表 3 的四组配置各提供 seed 100、101、102 的检查点：

| 配置 | ID | 文件 |
| --- | --- | --- |
| HiBlastNet | `hiblastnet` | `current_bc_lcjs_seed{100,101,102}.pth` |
| 细化对照（基线 + BC-MSRR） | `gcnet_bc` | `current_bc_control_seed{100,101,102}.pth` |
| 基线 + LCJS | `gcnet_lcjs` | `current_b_lcjs_seed{100,101,102}.pth` |
| 基线（GCNet-S） | `gcnet` | `current_b_plain_seed{100,101,102}.pth` |

| 网盘 | 链接 | 提取码 |
| --- | --- | --- |
| OneDrive | [下载](https://1drv.ms/u/c/1495c176035c0989/IQBf-shaXSRVTaecn952a2AjAYvkrAHUWIZh25kWocqEaow?e=8L8BUk) | — |
| 百度网盘 | [下载](https://pan.baidu.com/s/1EQbfUBzC3oSwfghtue5rhA?pwd=9zgt) | `9zgt` |

权重采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。解压后将 12 个 `.pth` 文件放入 `weights/`；压缩包里的 `manifest.json` 提供逐文件 SHA-256 校验值。评估请使用仓库的 `configs/evaluation/`，压缩包内的 `test_configs/` 记录的是原始实验环境。

发布的检查点已经过推理重参数化（`deploy` 格式）。其他对比模型与消融设置可按下文自行训练。检查点通过 Python pickle 加载，请只使用可信来源的权重。

## 训练

使用默认 seed 100 训练：

```bash
python train.py --config configs/training/hiblastnet.yaml
```

论文报告 seed 100、101、102 的平均结果。更换随机种子：

```bash
python train.py --config configs/training/hiblastnet.yaml --seed 101
```

每次训练在 `outputs/training/hiblastnet/gcnet_s_YYYYMMDD_HHMMSS/` 下保存训练曲线和检查点。`checkpoint_best.pth` 是验证集病斑 Dice 最佳的 EMA 权重，用于评估；`checkpoint_latest.pth` 用于续训。

训练共 140 轮，每轮取 2,000 个裁剪块，每 5 轮做一次整叶验证。续训时如曾更改随机种子，请继续传入相同的 `--seed`：

```bash
python train.py --config configs/training/hiblastnet.yaml --resume outputs/training/hiblastnet/RUN_FOLDER/checkpoint_latest.pth
```

显存不足可加 `--batch-size 2`。数据加载进程有问题可加 `--num-workers 0`。其他参数见 YAML 文件，其中的路径相对于配置文件所在目录。

## 评估

在测试集评估发布的 HiBlastNet 权重（默认 seed 100）：

```bash
python scripts/evaluate.py
```

其他种子用 `--seed 101` 或 `--seed 102`；其他已发布配置用 `--method gcnet_bc`、`gcnet_lcjs` 或 `gcnet`。输出目录为 `outputs/<method>/seed<N>/test_gcnet_s_full_leaf_<time>/`：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 汇总像素指标（病斑 Dice 为 `lesion_mean_dice`）、分尺寸组件指标、边界 F1、每叶假阳性组件数 |
| `per_leaf_metrics.csv` | 逐叶指标 |
| `component_metrics.csv` | 逐叶组件计数 |
| `full_leaf_predictions/` | 类别标签掩膜 |
| `full_leaf_visualizations/` | 原图、标注与预测对照图 |

评估自行训练且尚未重参数化的检查点时，传入文件路径并使用 `--deploy false`：

```bash
python scripts/evaluate.py --model-path outputs/training/hiblastnet/RUN_FOLDER/checkpoint_best.pth --deploy false
```

使用 `--list` 查看方法，`--split val` 评估验证集，`--dry-run` 仅打印实际执行的 `test.py` 命令。

**复现论文表 3。** 运行 `python scripts/check_reproduction.py` 评估 12 个发布检查点，并对照论文数值输出三个随机种子的均值和标准差。

**进一步分析。** `scripts/analyze_predictions.py` 读取保存的掩膜，计算 IoU 阈值 0.10 和 0.50 下的一对一组件匹配指标、逐叶病斑面积误差、去除小于 24 像素的预测组件后的结果，以及可选的按来源照片抽样的 bootstrap 区间。

```bash
python scripts/analyze_predictions.py --dataset-root data/blastleaf --split test \
    --predictions hiblastnet=outputs/hiblastnet/seed100 gcnet_bc=outputs/gcnet_bc/seed100 \
    --reference gcnet_bc --bootstrap 10000 --output outputs/analysis_seed100.json
```

## 单叶预测

输入与 BlastLeaf 一致的已标定 ROI：一张图一片叶子、60 像素/mm、长轴竖直。带标定板的完整照片请使用[原图评级](#原图评级)。

```bash
python full_leaf_inference.py --model-path weights/current_bc_lcjs_seed100.pth --image path/to/leaf.png --output-dir outputs/prediction
```

输出与输入尺寸相同的 0–3 类别标签 PNG（普通图片查看器中看起来接近全黑），以及记录滑窗位置的 `.tiles.json`。批量预测一个集合时，将 `--image` 替换为 `--dataset-root data/blastleaf --split test`。

## 原图评级

可用[原图示例](#原图示例)，也可以把离体叶片平铺在 [A4 ArUco 标定板](.github/media/reference-board-a4.svg) 上拍摄。按 **100% / 实际大小**打印模板，确认每个标记宽 16 mm。叶片不要重叠，照片须拍全两侧标记。该流程不适用于未使用标定板的任意田间照片。

安装基础依赖并将 `current_bc_lcjs_seed100.pth` 放入 `weights/` 后执行：

```bash
python -m pip install -r requirements-photo.txt
python scripts/run_photo_pipeline.py --image "path/to/board.jpg" --output-dir outputs/board
```

可用 `--model-path` 更换下载的权重，或用 `--device cpu` 强制在 CPU 上运行。输出目录须为新目录或空目录。

| 输出 | 内容 |
| --- | --- |
| `summary.csv` | 每片检出叶的等级、病斑计数、面积比例及长度 |
| `rectified_preview.jpg`、`leaf_locations.jpg` | 校正后的整板预览及编号定位图 |
| `leaves/leaf_*/` | 60 像素/mm 单叶图、类别标签掩膜、叠加图、几何信息和病斑测量 |
| `run.json` | 标定和推理设置，以及执行状态 |

脚本保留原始预测掩膜。测量时去除小于 24 像素的同类连通组件，再按论文的感病型病斑面积规则 `seedling_new_standard_v2` 计算 0–9 级。空掩膜不评级，标定不足时会报错。使用评级前请核对裁切和叠加图。

标定板布局和叶片检测参考配套的 `rice_blast_system`。定位在缩小的整板预览中进行，推理用的单叶图则以 60 像素/mm 直接从原图采样。

## 方法与配置

各方法的训练和评估配置分别位于 `configs/training/` 和 `configs/evaluation/`。[实验索引](configs/EXPERIMENTS.csv)列出了训练命令、评估配置和发布权重。

| ID | 配置 |
| --- | --- |
| `hiblastnet` | GCNet-S + BC-MSRR + LCJS（尺寸、局部、健康组织） |
| `gcnet_bc` | GCNet-S + BC-MSRR，不使用 LCJS |
| `gcnet_bc_size` | 细化对照 + LCJS 尺寸项 |
| `gcnet_bc_local_healthy` | 细化对照 + LCJS 局部项和健康项 |
| `gcnet_bc_size_local` | 细化对照 + LCJS 尺寸项和局部项 |
| `gcnet_bc_size_healthy` | 细化对照 + LCJS 尺寸项和健康项 |
| `gcnet` | GCNet-S 基线 |
| `gcnet_lcjs` | GCNet-S + LCJS，不使用 BC-MSRR |
| `gcnet_bc_blob` | GCNet-S + BC-MSRR + 改编的 Blob 损失 |
| `unet` | U-Net |
| `deeplabv3_resnet50` | DeepLabV3-ResNet50 |
| `pidnet_s` | PIDNet-S |
| `sctnet_b` | SCTNet-B |
| `efficientvit_b2` | EfficientViT-B2 |
| `segformer_b1` | SegFormer-B1 |

所有对比模型均从头训练。例如训练和评估 U-Net：

```bash
python train.py --config configs/training/unet.yaml
python scripts/evaluate.py --method unet --model-path outputs/training/unet/RUN_FOLDER/checkpoint_best.pth
```

Blob 损失配置要使用专用训练脚本；直接交给 `train.py` 会训练细化对照：

```bash
python scripts/train_blob_loss.py --config configs/training/gcnet_bc_blob.yaml --blob-weight 0.3
```

<details>
<summary>BC-MSRR 和 LCJS 在配置文件中的对应参数</summary>

| 论文组件 | `hiblastnet.yaml` 中的配置 |
| --- | --- |
| BC-MSRR 分支 | `model.gcnet_multiscale_detail_refinement: true`、`gcnet_multiscale_context_branch: true`、`gcnet_multiscale_boundary_fusion: true` |
| BC-MSRR 边界损失 | `loss.gcnet_refinement_boundary_weight: 0.25` |
| LCJS 尺寸项 | `loss.gcnet_small_instance_weight: 0.1`，并使用 `_alpha`、`_tau`、`_max_weight` |
| LCJS 组件局部项 | `loss.gcnet_cpmr_direct_weight: 0.02`，并使用 `gcnet_cpmr_context_pixels`、`_max_components`、`_small_area`、`_small_fraction`、`_dice_weight` |
| LCJS 健康组织难负样本项 | `loss.gcnet_cpmr_healthy_weight: 0.01` |

组件局部项和健康组织项在前 3 轮逐渐增大权重，尺寸项从训练开始就使用完整权重。

</details>

## 复现说明

- 论文报告三个种子（100、101、102）的均值 ± 标准差。单次运行会有差异，硬件和软件版本也可能影响结果。
- 按验证集上 EMA 权重的病斑 Dice 选取检查点，测试集不参与训练。
- 仓库是论文代码的整理版。`scripts/check_reproduction.py` 对照论文表 3 验证发布的权重；同一检查点重新推理时，部分数值的末位可能有变化（见论文补充材料 S5）。

## 引用

论文正在准备中。正式引用信息可用后会补充 BibTeX，现阶段可引用本仓库的网址。

## 许可证

| 内容 | 许可证 |
| --- | --- |
| 本仓库代码 | [MIT](LICENSE) |
| 第三方代码 | 沿用各自原许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [licenses/](licenses/) |
| BlastLeaf 数据集 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 预训练权重 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

SegFormer 文件（`blastseg/models/segformer*.py`）遵循 NVIDIA Source Code License，仅可用于非商业研究或评估。如需将代码用于商业用途，请移除 SegFormer 对比模型。
