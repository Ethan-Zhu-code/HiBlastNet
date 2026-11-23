# HiBlastNet usage guide

[Project home](../README.md) · [简体中文](usage.zh-CN.md)

Follow the [installation instructions](../README.md#install) and [download the files](../README.md#downloads) for your task. Run all commands from the repository root.

| Keep leaf detail | Refine lesion boundaries | Supervise lesion components |
| :--- | :--- | :--- |
| Leaves are processed at their calibrated resolution in overlapping 1024 × 640 windows, and the window logits are fused with Hann weights. | BC-MSRR adds a refinement branch that combines local, contextual and boundary features. | LCJS adds size-aware, component-local and healthy-tissue terms to the training loss. It adds nothing at inference time. |

**Scope.** The repository covers the segmentation experiments and a command-line workflow for reference-board photographs: board rectification, leaf extraction, full-leaf segmentation, and the operational 0–9 grade. The desktop application is separate.

## Dataset

The **BlastLeaf** dataset contains 707 leaf regions of interest (ROIs) from 283 photographs of detached rice leaves, with pixel-level labels.

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

### Original photo examples

Download **`BlastLeaf_30_Original_Photos.zip`** (419 MB) for 30 original reference-board JPEGs from the BlastLeaf test split. The photos correspond to 72 annotated leaf ROIs and are provided separately from the cropped ROI dataset. The archive includes a manifest with source image IDs, original filenames, and SHA-256 checksums.

## Trained weights

Checkpoints are released for the four configurations of the paper's ablation (Table 7), each trained with seeds 100, 101 and 102:

| Configuration | ID | Files |
| --- | --- | --- |
| HiBlastNet | `hiblastnet` | `current_bc_lcjs_seed{100,101,102}.pth` |
| Refinement control (Baseline + BC-MSRR) | `gcnet_bc` | `current_bc_control_seed{100,101,102}.pth` |
| Baseline + LCJS | `gcnet_lcjs` | `current_b_lcjs_seed{100,101,102}.pth` |
| Baseline (GCNet-S) | `gcnet` | `current_b_plain_seed{100,101,102}.pth` |

The weights are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Put the twelve `.pth` files in `weights/`. The archive's `manifest.json` lists the SHA-256 checksum of every file. Use the configurations in `configs/evaluation/` rather than the archive's `test_configs/`, which record the original training environment.

The released checkpoints are re-parameterized for inference ("deploy" form). The comparison models and the other ablations are not released; train them with the commands below. Checkpoints are loaded with Python pickle, so only load checkpoints from sources you trust.

<details>
<summary>SHA-256 checksums of all 12 released files</summary>

The checksums below were computed from the checkpoint bytes inside `HiBlastNet_Weights.zip` and match its `manifest.json`. The six HiBlastNet and refinement-control hashes also match the values supplied for paper Dataset S1; the seed-100 HiBlastNet hash starts with `9d3f5da2`, as recorded in Dataset S2. The other six hashes identify the release files and were not independently compared with a paper hash in this audit.

| File (under `weights/`) | SHA-256 |
| --- | --- |
| `current_bc_lcjs_seed100.pth` | `9d3f5da2ef7117a7a679455498c4acb2d92ee61f3a488af4491f07911659bb10` |
| `current_bc_lcjs_seed101.pth` | `d828b1d41749b01c4bd674da2d342e1130d02c1d4318cac07b014db1a5c62998` |
| `current_bc_lcjs_seed102.pth` | `9b49f3dd769db333f80eb0ea229f12eb419ab5ffd793756d3057976ee79c17ad` |
| `current_bc_control_seed100.pth` | `9eda19af56462e83a41df0be845c0b6625f754a55d3f31bb2f122f9f631a91c1` |
| `current_bc_control_seed101.pth` | `47e2e24636e3863cb97bfed93aa86b7ab49d72e4387ffb332ca4fc7780c50d64` |
| `current_bc_control_seed102.pth` | `07d5e3a4f3ab9f57f1b6286b36249bbc025fb1efa7877dfe696ff505a167457b` |
| `current_b_lcjs_seed100.pth` | `d59bcf43e308619c91d6f39c995a60b2227fbbd90ff8ec76ba35d05ef81f7e95` |
| `current_b_lcjs_seed101.pth` | `7ee214cdc9be93e9b92eb77fba63927c57cdce86dab0c2d89ba8de19f0fe0684` |
| `current_b_lcjs_seed102.pth` | `614b90310b076d05fee249c60e4bd5cedd9a6207f139377027a1b484ae0c297d` |
| `current_b_plain_seed100.pth` | `9a11a5553d76307c48e4cb6d1404e13c5124d0764d58f6f5a79f774996d6a574` |
| `current_b_plain_seed101.pth` | `c2d8f0b70123cfcfae6a40077f57c87c31a4327631f276c68f8e65a294006d05` |
| `current_b_plain_seed102.pth` | `94544aeab444b3c939c7778731b72e2c2acf2d6755705216a1743f2c87e77087` |

</details>

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

**Reproduce Table 7.** `python scripts/check_reproduction.py` evaluates the twelve released checkpoints and prints mean ± SD over the three seeds next to the paper's values.

The references are the three-seed mean ± sample SD from the original, unfiltered ablation predictions in Table 7. Matched recall uses a component IoU threshold of 0.10. The default `--tolerance 0.05` checks the absolute difference between the reproduced three-seed mean and the paper mean: **0.05 percentage points** for Dice, Small survival, Matched recall, and Boundary F1, and **0.05 components per leaf** for FP/leaf. The sample SD is displayed but is not checked against a tolerance; there is no per-seed reference-value check. Re-inference of the same weights can differ in the last reported digit (Supplementary Section S5).

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

This entry point takes calibrated leaf ROIs like those in BlastLeaf: one leaf per image, 60 px/mm, long axis vertical. For an original reference-board photograph, use [Grade a raw photo](#grade-a-raw-photo).

```bash
python full_leaf_inference.py --model-path weights/current_bc_lcjs_seed100.pth --image path/to/leaf.png --output-dir outputs/prediction
```

The output is a class-label PNG at the input size, using labels 0–3 (it looks black in ordinary image viewers), plus a `.tiles.json` file describing the windows. To predict every leaf of a dataset split, use `--dataset-root data/blastleaf --split test` instead of `--image`.

## Grade a raw photo

Use one of the [original photo examples](#original-photo-examples), or take a photo of detached leaves on the [A4 ArUco reference board](../.github/media/reference-board-a4.svg). Print the template at **100% / actual size**, check that each marker is 16 mm wide, and lay the leaves flat in the central area without overlap. Keep the markers in view on both sides. Arbitrary field photos without this board are outside this workflow.

After installing the main requirements and putting `current_bc_lcjs_seed100.pth` in `weights/`, run:

```bash
python -m pip install -r requirements-photo.txt
python scripts/run_photo_pipeline.py --image "path/to/board.jpg" --output-dir outputs/board
```

Use `--model-path` to select another downloaded checkpoint and `--device cpu` to force CPU inference. The output directory must be new or empty.

| Output | What it contains |
| --- | --- |
| `summary.csv` | One row per detected leaf: grade, lesion counts, fractions, and lengths |
| `rectified_preview.jpg`, `leaf_locations.jpg` | Rectified board and numbered crop locations for review |
| `leaves/leaf_*/` | 60 px/mm ROI, class-label mask, colour overlay, geometry and lesion measurements |
| `run.json` | Calibration, inference settings, and completion or failure status |

The script retains the raw prediction mask. For measurements, it excludes same-class components smaller than 24 pixels and applies the paper's susceptible-area-based 0–9 rule (`seedling_new_standard_v2`). An empty predicted leaf mask is left ungraded. No physical-area minimum is applied. Dice, Small survival, and the other Table 7 segmentation metrics use unfiltered masks; the 24-pixel filter is for reporting and grading. Check the crop and overlay before using the grade. A photo with inadequate calibration fails rather than assigning a grade at an assumed scale.

The board layout and ROI detector are adapted from the companion `rice_blast_system` implementation. Detection uses a reduced board preview; each inference crop is sampled directly from the source image at 60 px/mm.

## Methods and configurations

Each method has one training configuration in `configs/training/` and one evaluation configuration in `configs/evaluation/`. [`configs/EXPERIMENTS.csv`](../configs/EXPERIMENTS.csv) lists the training command, evaluation configuration and released checkpoint for each method.

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
- This code is a cleaned version of the code used for the paper. `scripts/check_reproduction.py` compares the released checkpoints with Table 7. Table 7 comes from the original inference run; a later re-inference of the same checkpoints changed a few values in the last digit (paper, Supplementary S5).
