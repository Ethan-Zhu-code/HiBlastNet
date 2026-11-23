<p align="center"><strong>English</strong> &nbsp;·&nbsp; <a href="README.zh-CN.md">简体中文</a></p>

<p align="center">
  <img src=".github/media/hiblastnet-banner-clean.png" alt="HiBlastNet — rice leaf blast lesion segmentation. Botanical leaf and segmentation concept illustration." width="100%" />
</p>

# HiBlastNet

HiBlastNet segments susceptible-type and resistant-type rice blast lesions. It processes each leaf in overlapping windows at 60 pixels per millimetre to preserve small lesions that can disappear when the whole leaf is resized. This repository includes training and evaluation code and a command-line tool for extracting leaves from reference-board photographs, measuring lesions and assigning a 0–9 severity grade. The desktop application is maintained separately.

## Install

Install Git and Miniconda, then run:

```bash
git clone https://github.com/Ethan-Zhu-code/HiBlastNet.git
cd HiBlastNet
conda create -n hiblastnet python=3.11 -y
conda activate hiblastnet
```

Install PyTorch and torchvision with the command for your hardware from the [PyTorch installation page](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

The paper experiments used PyTorch 2.5.1 with CUDA 12.1. Training needs an NVIDIA GPU. Evaluation and prediction also run on CPU, slowly.

## Downloads

| Resource | Contents | OneDrive | Baidu and extraction code |
| --- | --- | --- | --- |
| BlastLeaf dataset | 707 leaves from 283 photographs, with 71,255 annotated lesion components | [OneDrive](https://1drv.ms/u/c/1495c176035c0989/IQCrwwXhApP_TpZNFzTIXQelAe2YA5BwKQ68frvLGekOpbQ?e=dtETWr) | [Baidu](https://pan.baidu.com/s/1K6AEge89TWjPSs5MAovVrg?pwd=p1p6) · `p1p6` |
| Reference-board photos | 30 original test photographs covering 72 annotated leaves (419 MB) | [OneDrive](https://1drv.ms/u/c/1495c176035c0989/IQBtNABm1JC3SIWQSa1nlXPRAYTeY_4OhdFh1d5rHNjfLeA?e=g9yzBJ) | [Baidu](https://pan.baidu.com/s/1EUgH1kOioeeU1JFz02Q_cw?pwd=ek4a) · `ek4a` |
| Trained weights | 12 checkpoints covering four configurations and three seeds | [OneDrive](https://1drv.ms/u/c/1495c176035c0989/IQC_DSvX2FfxRrrIhwz_O7L-AfOekbesJ90LgyT1dyQLT-o?e=HRmyBq) | [Baidu](https://pan.baidu.com/s/1EQbfUBzC3oSwfghtue5rhA?pwd=9zgt) · `9zgt` |

## Usage

[Prepare data and weights](docs/usage.md#dataset) · [Train](docs/usage.md#train) · [Evaluate and reproduce results](docs/usage.md#evaluate) · [Predict a leaf mask](docs/usage.md#predict) · [Grade a board photograph](docs/usage.md#grade-a-raw-photo)

## Citation

*HiBlastNet: a scale-preserving segmentation network for small rice blast lesions in quantitative measurement and severity grading* is in preparation. A BibTeX entry will be added here once it is available. Until then, please cite this repository by its URL.

## License

| Part | License |
| --- | --- |
| Code in this repository | [MIT](LICENSE) |
| Third-party code | Its original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` |
| BlastLeaf dataset | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Trained weights | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The SegFormer files (`blastseg/models/segformer*.py`) are under the NVIDIA Source Code License and may be used only for non-commercial research or evaluation. Remove the SegFormer comparison model if you need to use the code commercially.
