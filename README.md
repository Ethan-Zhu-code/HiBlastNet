<p align="center">
  <img src=".github/media/hiblastnet-banner.png" alt="HiBlastNet — rice leaf blast lesion segmentation. Botanical leaf and segmentation concept illustration." width="100%" />
</p>

<p align="center">
  <strong>Full-leaf rice blast lesion segmentation on calibrated leaf images.</strong><br />
  GCNet-S backbone · BC-MSRR refinement · LCJS supervision
</p>

<p align="center">
  <a href="#install">Install</a> &nbsp;·&nbsp;
  <a href="#dataset">Dataset</a> &nbsp;·&nbsp;
  <a href="#trained-weights">Weights</a> &nbsp;·&nbsp;
  <a href="#train">Train</a> &nbsp;·&nbsp;
  <a href="#evaluate">Evaluate</a> &nbsp;·&nbsp;
  <a href="#predict">Predict</a> &nbsp;·&nbsp;
  <a href="#methods-and-configurations">Methods</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

---

This repository contains the code to train and evaluate the segmentation models in the paper *Preserving small lesions in full-leaf segmentation of rice blast: a calibrated workflow for lesion morphology and severity measurement*. HiBlastNet labels every pixel of a leaf image as background, healthy tissue, susceptible-type lesion or resistant-type lesion.

| Keep leaf detail | Refine lesion boundaries | Supervise lesion components |
| :--- | :--- | :--- |
| Leaves are processed at their calibrated resolution in overlapping 1024 × 640 windows, and the window logits are fused with Hann weights. | BC-MSRR adds a refinement branch that combines local, contextual and boundary features. | LCJS adds size-aware, component-local and healthy-tissue terms to the training loss. It adds nothing at inference time. |

**Scope.** The repository covers the segmentation experiments: training, full-leaf evaluation and prediction for HiBlastNet, its ablations and the comparison models. It does **not** include the steps around them that the paper also describes:

- reference-board rectification and leaf-ROI extraction from raw photographs;
- the lesion descriptors, the 0–9 operational grade and the desktop application.

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

The paper experiments used PyTorch 2.5.1 with CUDA 12.1. Training needs an NVIDIA GPU. Evaluation and prediction also run on CPU, slowly. Run all commands from the `HiBlastNet` folder.

## Dataset

The **BlastLeaf** dataset contains 707 leaf regions of interest (ROIs) from 283 photographs of detached rice leaves, with pixel-level labels. Download it from either mirror:

| Mirror | Link | Extraction code |
| --- | --- | --- |
| OneDrive | [Download](https://1drv.ms/u/c/1495c176035c0989/IQC3HxyG_ktHSpHlq_E-5vmXAYjrLFhmseHouaPyHpjrC-4?e=SkNipz) | — |
| Baidu Netdisk | [Download](https://pan.baidu.com/s/1K6AEge89TWjPSs5MAovVrg?pwd=p1p6) | `p1p6` |

BlastLeaf is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): you may share and adapt it for any purpose, including commercially, provided you give appropriate credit.

Extract the archive and rename the extracted folder so that the files sit under `data/blastleaf/`:

```text
HiBlastNet/
  data/blastleaf/
    manifest.csv
    split_summary.json
    train/   images/  masks/  relations/
    val/     images/  masks/  relations/
    test/    images/  masks/  relations/
```

**Images.** Each image is one leaf ROI cut from a photograph that was rectified with a reference board to 60 pixels per mm, with the leaf's long axis vertical.

**Masks.** Masks are single-channel PNGs with these labels:

| Label | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Healthy leaf tissue, including damage not caused by blast |
| 2 | Susceptible-type lesion |
| 3 | Resistant-type lesion |

The lesion classes describe lesion appearance, not the resistance of the plant.

**Split.** Photographs, not leaves, were assigned to the training, validation and test sets, so leaves from one photograph always share a set. The sets contain 503, 104 and 100 leaves. Keep this split when comparing results.

**`manifest.csv`.** The manifest lists one leaf per row with these columns:

- `sample_id`, `split`, `source_image_id` and `leaf_id`;
- `image_path`, `mask_path` and `relation_path`, relative to the dataset root;
- `width` and `height`;
- the four per-class pixel counts.

`split_summary.json` records the photograph-to-split assignment. Each `relations/*.json` file repeats a leaf's identity, size and file paths. The loader checks all of these files for consistency before training or evaluation. To train on your own data, produce the same structure (see `blastseg/data/full_leaf.py`).

## Trained weights

Checkpoints are released for the four configurations of the paper's ablation (Table 3), each trained with seeds 100, 101 and 102:

| Configuration | ID | Files |
| --- | --- | --- |
| HiBlastNet | `hiblastnet` | `current_bc_lcjs_seed{100,101,102}.pth` |
| Refinement control (Baseline + BC-MSRR) | `gcnet_bc` | `current_bc_control_seed{100,101,102}.pth` |
| Baseline + LCJS | `gcnet_lcjs` | `current_b_lcjs_seed{100,101,102}.pth` |
| Baseline (GCNet-S) | `gcnet` | `current_b_plain_seed{100,101,102}.pth` |

| Mirror | Link | Extraction code |
| --- | --- | --- |
| OneDrive | [Download](https://1drv.ms/u/c/1495c176035c0989/IQBf-shaXSRVTaecn952a2AjAYvkrAHUWIZh25kWocqEaow?e=8L8BUk) | — |
| Baidu Netdisk | [Download](https://pan.baidu.com/s/1EQbfUBzC3oSwfghtue5rhA?pwd=9zgt) | `9zgt` |

The weights are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Put the twelve `.pth` files in `weights/`. The archive's `manifest.json` lists the SHA-256 checksum of every file. Use the configurations in `configs/evaluation/` rather than the archive's `test_configs/`, which record the original training environment.

The released checkpoints are re-parameterized for inference ("deploy" form). The comparison models and the other ablations are not released; train them with the commands below. Checkpoints are loaded with Python pickle, so only load checkpoints from sources you trust.

## Train

Train HiBlastNet with the default seed 100:

```bash
python train.py --config configs/training/hiblastnet.yaml
```

The paper reports the mean over seeds 100, 101 and 102. To train with another seed:

```bash
python train.py --config configs/training/hiblastnet.yaml --seed 101
```

Each run writes a new folder `outputs/training/hiblastnet/gcnet_s_YYYYMMDD_HHMMSS/` containing:

- `checkpoint_best.pth`, the EMA weights with the best validation lesion Dice, which is the checkpoint to evaluate;
- `checkpoint_latest.pth`, for resuming;
- the training history and plots.

Training runs 140 epochs of 2,000 crops each, with full-leaf validation every 5 epochs. To resume an interrupted run (pass the same `--seed` again if you changed it):

```bash
python train.py --config configs/training/hiblastnet.yaml --resume outputs/training/hiblastnet/RUN_FOLDER/checkpoint_latest.pth
```

If GPU memory is short, add `--batch-size 2`. If data-loader workers cause problems, add `--num-workers 0`. Other settings are in the YAML files; paths inside a YAML file are relative to that file.

## Evaluate

Evaluate the released HiBlastNet checkpoint (seed 100) on the test set:

```bash
python scripts/evaluate.py
```

Add `--seed 101` or `--seed 102` for the other runs, and `--method gcnet_bc`, `gcnet_lcjs` or `gcnet` for the other released configurations. Results go to `outputs/<method>/seed<N>/test_gcnet_s_full_leaf_<time>/`:

| File | Contents |
| --- | --- |
| `metrics.json` | Pooled pixel metrics (lesion Dice is `lesion_mean_dice`), component metrics by size bin, boundary F1 and false-positive components per leaf |
| `per_leaf_metrics.csv` | Per-leaf scores |
| `component_metrics.csv` | Per-leaf component counts |
| `full_leaf_predictions/` | Predicted class-label masks |
| `full_leaf_visualizations/` | Image, reference and prediction panels |

To evaluate a checkpoint from your own training run, pass its path and `--deploy false`, because training checkpoints are not re-parameterized:

```bash
python scripts/evaluate.py --model-path outputs/training/hiblastnet/RUN_FOLDER/checkpoint_best.pth --deploy false
```

Other options:

- `python scripts/evaluate.py --list` lists all methods.
- `--method <id>` evaluates another method.
- `--split val` evaluates the validation set.
- `--dry-run` prints the underlying `test.py` command without running it.

**Reproduce Table 3.** `python scripts/check_reproduction.py` evaluates the twelve released checkpoints and prints mean ± SD over the three seeds next to the paper's values.

**Further analyses.** `scripts/analyze_predictions.py` reads saved prediction masks and computes the remaining analyses in the paper:

- one-to-one component recall, precision and F1 at IoU ≥ 0.10 and ≥ 0.50, by lesion size;
- leaf-level lesion-area agreement (MAE, RMSE and bias);
- the same metrics after removing predicted components smaller than 24 pixels;
- optional source-photograph bootstrap intervals, including paired differences between models.

```bash
python scripts/analyze_predictions.py --dataset-root data/blastleaf --split test \
    --predictions hiblastnet=outputs/hiblastnet/seed100 gcnet_bc=outputs/gcnet_bc/seed100 \
    --reference gcnet_bc --bootstrap 10000 --output outputs/analysis_seed100.json
```

## Predict

Prediction expects calibrated leaf ROIs like those in BlastLeaf: one leaf per image, 60 px/mm, long axis vertical. Raw photographs are not supported, because the ROI extraction step is not part of this repository.

```bash
python full_leaf_inference.py --model-path weights/current_bc_lcjs_seed100.pth --image path/to/leaf.png --output-dir outputs/prediction
```

The output is a class-label PNG at the input size, using labels 0–3 (it looks black in ordinary image viewers), plus a `.tiles.json` file describing the windows. To predict every leaf of a dataset split, use `--dataset-root data/blastleaf --split test` instead of `--image`.

## Methods and configurations

Each method has one training configuration in `configs/training/` and one evaluation configuration in `configs/evaluation/`. [`configs/EXPERIMENTS.csv`](configs/EXPERIMENTS.csv) lists the training command, evaluation configuration and released checkpoint for each method.

| ID | Setting |
| --- | --- |
| `hiblastnet` | GCNet-S + BC-MSRR + LCJS (size, local, healthy) |
| `gcnet_bc` | Refinement control: GCNet-S + BC-MSRR without LCJS |
| `gcnet_bc_size` | + LCJS size term |
| `gcnet_bc_local_healthy` | + LCJS local and healthy terms |
| `gcnet_bc_size_local` | + LCJS size and local terms |
| `gcnet_bc_size_healthy` | + LCJS size and healthy terms |
| `gcnet` | Baseline: GCNet-S without BC-MSRR or LCJS |
| `gcnet_lcjs` | GCNet-S + LCJS without BC-MSRR |
| `gcnet_bc_blob` | GCNet-S + BC-MSRR + adapted Blob loss; train with `scripts/train_blob_loss.py` (see below) |
| `unet` | U-Net |
| `deeplabv3_resnet50` | DeepLabV3-ResNet50 |
| `pidnet_s` | PIDNet-S |
| `sctnet_b` | SCTNet-B |
| `efficientvit_b2` | EfficientViT-B2 |
| `segformer_b1` | SegFormer-B1 |

All models are trained from scratch. For example, to train and evaluate U-Net:

```bash
python train.py --config configs/training/unet.yaml
python scripts/evaluate.py --method unet --model-path outputs/training/unet/RUN_FOLDER/checkpoint_best.pth
```

**Blob loss.** The Blob-loss model needs the separate training script; running `train.py` with its configuration would train the refinement control instead:

```bash
python scripts/train_blob_loss.py --config configs/training/gcnet_bc_blob.yaml --blob-weight 0.3
```

<details>
<summary>Where BC-MSRR and LCJS appear in the configuration files</summary>

The configuration keys keep the names used during development.

| Paper component | Configuration keys (values in `hiblastnet.yaml`) |
| --- | --- |
| BC-MSRR branch | `model.gcnet_multiscale_detail_refinement: true`, `gcnet_multiscale_context_branch: true`, `gcnet_multiscale_boundary_fusion: true` |
| BC-MSRR boundary loss | `loss.gcnet_refinement_boundary_weight: 0.25` |
| LCJS size term | `loss.gcnet_small_instance_weight: 0.1` (with `_alpha`, `_tau`, `_max_weight`) |
| LCJS component-local term | `loss.gcnet_cpmr_direct_weight: 0.02` (with `gcnet_cpmr_context_pixels`, `_max_components`, `_small_area`, `_small_fraction`, `_dice_weight`) |
| LCJS healthy hard-negative term | `loss.gcnet_cpmr_healthy_weight: 0.01` |

The component-local and healthy terms are ramped up over the first three epochs. The size term is applied at full weight from the start.

</details>

## Reproduction notes

- The paper reports mean ± standard deviation over seeds 100, 101 and 102. A single run will differ from that mean; scores also vary with hardware and software versions.
- Checkpoints are selected on the validation set by lesion Dice using the EMA weights. The test set is not used during training.
- This code is a cleaned version of the code used for the paper. `scripts/check_reproduction.py` compares the released checkpoints with Table 3. Table 3 comes from the original inference run; a later re-inference of the same checkpoints changed a few values in the last digit (paper, Supplementary S5).

## Citation

The paper describing HiBlastNet and BlastLeaf is in preparation. A BibTeX entry will be added here once it is available. Until then, please cite this repository by its URL.

## License

| Part | License |
| --- | --- |
| Code in this repository | [MIT](LICENSE) |
| Third-party code | Its original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` |
| BlastLeaf dataset | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Trained weights | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The SegFormer files (`blastseg/models/segformer*.py`) are under the NVIDIA Source Code License and may be used only for non-commercial research or evaluation. Remove the SegFormer comparison model if you need to use the code commercially.
