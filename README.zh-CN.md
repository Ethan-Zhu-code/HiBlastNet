<p align="center"><a href="README.md">English</a> &nbsp;·&nbsp; <strong>简体中文</strong></p>

<p align="center">
  <img src=".github/media/hiblastnet-banner-clean.png" alt="HiBlastNet 水稻叶瘟病斑分割示意图" width="100%" />
</p>

# HiBlastNet

HiBlastNet 用于分割水稻叶瘟的小病斑。模型在标定分辨率下逐窗处理整片叶子，区分健康叶组织、感病型病斑和抗病型病斑。对于使用标定板拍摄的照片，仓库提供的命令行工具可以逐叶提取病斑数量、面积占比和长度，并给出 0–9 级病情评分。

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

论文实验使用 PyTorch 2.5.1 和 CUDA 12.1。训练需要 NVIDIA GPU；评估和预测也可在 CPU 上运行，但速度较慢。

## 下载

| 资源 | 内容 | OneDrive | 百度网盘 · 提取码 |
| --- | --- | --- | --- |
| BlastLeaf 数据集 | 283 张照片中的 707 片水稻叶片，含像素级掩膜 | [OneDrive](https://1drv.ms/u/c/1495c176035c0989/IQCrwwXhApP_TpZNFzTIXQelAe2YA5BwKQ68frvLGekOpbQ?e=dtETWr) | [百度网盘](https://pan.baidu.com/s/1K6AEge89TWjPSs5MAovVrg?pwd=p1p6) · `p1p6` |
| 标定板原始照片 | 30 张测试集 JPEG（419 MB），对应 72 片已标注叶片 | [OneDrive](https://1drv.ms/u/c/1495c176035c0989/IQBtNABm1JC3SIWQSa1nlXPRAYTeY_4OhdFh1d5rHNjfLeA?e=g9yzBJ) | [百度网盘](https://pan.baidu.com/s/1EUgH1kOioeeU1JFz02Q_cw?pwd=ek4a) · `ek4a` |
| 模型权重 | 4 组配置，每组 3 个随机种子，共 12 个检查点 | [OneDrive](https://1drv.ms/u/c/1495c176035c0989/IQC_DSvX2FfxRrrIhwz_O7L-AfOekbesJ90LgyT1dyQLT-o?e=HRmyBq) | [百度网盘](https://pan.baidu.com/s/1EQbfUBzC3oSwfghtue5rhA?pwd=9zgt) · `9zgt` |

将数据集解压到 `data/blastleaf/`，将 `.pth` 权重放入 `weights/`。拍摄新样本可使用仓库中的 [A4 标定板模板](.github/media/reference-board-a4.svg)。

## 使用指南

[训练](docs/usage.zh-CN.md#训练) · [评估](docs/usage.zh-CN.md#评估) · [单叶预测](docs/usage.zh-CN.md#单叶预测) · [原图评级](docs/usage.zh-CN.md#原图评级) · [方法与配置](docs/usage.zh-CN.md#方法与配置)

## 引用

论文 *HiBlastNet: a scale-preserving segmentation network for small rice blast lesions in quantitative measurement and severity grading* 正在准备中，现阶段请引用[本仓库](https://github.com/Ethan-Zhu-code/HiBlastNet)。

## 许可证

| 内容 | 许可证 |
| --- | --- |
| 本仓库代码 | [MIT](LICENSE) |
| 第三方代码 | 沿用各自原许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [licenses/](licenses/) |
| BlastLeaf 数据集 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| 预训练权重 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

SegFormer 文件（`blastseg/models/segformer*.py`）遵循 NVIDIA Source Code License，仅可用于非商业研究或评估。如需将代码用于商业用途，请移除 SegFormer 对比模型。
