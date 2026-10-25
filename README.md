# Simulated MRI Artifact Stress Test

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23645383.svg)](https://doi.org/10.5281/zenodo.23645383)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![MONAI](https://img.shields.io/badge/built%20with-MONAI-green.svg)

Code of the paper **"What Clean Dice Hides: Severity-Matched Stress Testing of 3D Brain Tumor Segmentation under
Simulated MRI Artifacts"**. The paper link and its citation will be added here on publication.

A mean Dice score on clean MRI says little about what happens to an individual patient when the scan is degraded. This
repository trains three 3D segmentation models, degrades the held-out test volumes with three physically motivated
artifacts at ten severities of **equal structural similarity (SSIM 0.95 to 0.50)**, and measures what the mean Dice
conceals: individual-level operating limits, boundary damage, tumor-volume change and the sources of variation.

| | |
|---|---|
| **Data** | Medical Segmentation Decathlon *Task01 BrainTumour* (BraTS), FLAIR only, whole tumor. 339 train / 48 validation / 97 test cases |
| **Models** | DynUNet, UNETR, SwinUNETR (MONAI), trained from scratch, 3 seeds each = 9 checkpoints |
| **Artifacts** | k-space **motion**, **Rician noise**, polynomial **bias field**; levels L1 to L10 (L1 to L5 primary, L6 to L10 exploratory) |
| **Scale** | 9 checkpoints x 97 test cases x 31 conditions (clean + 3 artifacts x 10 levels) = **27,063 predictions** |
| **Metrics** | DSC, IoU, HD95, normalized surface Dice (NSD), predicted tumor volume |

> **Outputs (weights, predictions, results, analysis tables) are not stored in this repository.**
> They are archived on Zenodo: **[doi.org/10.5281/zenodo.23645383](https://doi.org/10.5281/zenodo.23645383)**.
> Download the single zip (`simulated-mri-artifact-stress-test_Output.zip`) and extract it in the repository root, so
> that the folder `simulated-mri-artifact-stress-test_Output/` exists. This is enough to rerun the whole analysis
> without a GPU and without retraining (see [Reuse the Zenodo outputs](#8-reuse-the-zenodo-outputs)).

### Quick start (analysis only, no GPU)

```bash
git clone https://github.com/sarang-banakhede/Simulated-MRI-Artifact-Stress-Test.git
cd Simulated-MRI-Artifact-Stress-Test
python -m venv .venv && source .venv/bin/activate
pip install -e .
# download and extract the Zenodo zip here, then:
simulated-mri-artifact-stress-test analyze
```

### Contents

1. [Pipeline](#1-pipeline)
2. [Repository layout](#2-repository-layout)
3. [Install](#3-install)
4. [Input data](#4-input-data)
5. [Run](#5-run)
6. [Configuration](#6-configuration-configsconfigyaml)
7. [Outputs](#7-outputs)
8. [Reuse the Zenodo outputs](#8-reuse-the-zenodo-outputs)
9. [Reproducibility notes](#9-reproducibility-notes)
10. [Citation](#10-citation)
11. [License](#11-license)

---

## 1. Pipeline

```
prepare ──> train ──> calibrate ──> infer ──> analyze          reader-study (separate, optional)
  │           │           │           │          │
  │           │           │           │          └─ statistics, operating limits, volumetry, variance,
  │           │           │           │             mask-based analysis (error types, agreement, ensembles)
  │           │           │           └─ 31 conditions x 97 test cases x every checkpoint
  │           │           └─ artifact strength that gives the target SSIM of each level (48 validation cases)
  │           └─ 3 models x 3 seeds, resumable, best checkpoint by validation DSC
  └─ patient-level split + preprocessed training cache
```

Each stage is a command of the `simulated-mri-artifact-stress-test` CLI, can be run alone, and is **safe to rerun**:
finished work is skipped, and interrupted training and inference resume where they stopped.

## 2. Repository layout

```
configs/config.yaml          the only configuration file (paths, split, models, training, artifacts, inference, analysis)
src/simulated_mri_artifact_stress_test/
  cli.py                     command line: prepare | train | calibrate | infer | analyze | reader-study | run-all | status
  __main__.py                same CLI as `python -m simulated_mri_artifact_stress_test`
  config.py                  YAML loader, --set overrides, every output path of the pipeline
  utils.py                   logging, atomic file writes, time budget
  data/
    nifti.py                 NIfTI loading, brain-mask crop, z-score normalization inside the mask
    splits.py                patient-level split stratified by tumor volume, case_split.csv
    training_cache.py        preprocessed .npy cache for fast training
  models/factory.py          DynUNet, UNETR and SwinUNETR built from the YAML
  training/trainer.py        AdamW, warm-up + cosine schedule, Dice + BCE loss, validation, resume
  artifacts/
    simulate.py              motion, Rician noise, bias field, SSIM (deterministic pattern per case and artifact)
    calibration.py           SSIM-matched strengths, frozen-parameter check, frequency footprint
  evaluation/
    metrics.py               DSC, IoU, HD95, NSD, volumes
    inference.py             sliding-window inference over all conditions, masks, per-prediction results, reproduction check
  analysis/
    common.py                shared array layout (model x metric x artifact -> seed x level x case) and helpers
    results.py               merge of all results, per-checkpoint summary
    statistics.py            clean-Dice vs robustness (H1), pairwise reversals (H2), curves, tail risk, size, ranks
    operating.py             operating envelopes, clinical volumetry (ICC, limits of agreement), variance decomposition
    post_masks.py            mask-based analysis: error type, fragmentation, seed/model agreement, ensembles
    sensitivity.py           SSIM-restricted and adjusted analyses, pass-share sensitivity, relative-loss variance
  reader_study.py            clean and degraded volumes, blinded copy and slice overview for a radiologist
tests/                       20 fast unit tests and one end-to-end test on a synthetic dataset
pyproject.toml               package, dependencies and the CLI entry point
CITATION.cff                 citation metadata (used by GitHub's "Cite this repository" button)
LICENSE                      MIT license of the code
```

## 3. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                    # 20 fast unit tests, no data and no GPU needed
pytest -m slow               # end-to-end run of the whole pipeline on a tiny synthetic dataset (CPU, about a minute)
```

Requirements: Python 3.10 or newer. Training and inference need a CUDA GPU (a 4 GB GPU works with
`--set inference.sw_batch_size=1`); `analyze` runs on CPU.

Versions used for the paper runs: PyTorch 2.11, MONAI 1.6.1, NumPy 2.1, SciPy 1.16, scikit-image 0.25.

## 4. Input data

The pipeline **expects the data already arranged** in the structure below; it does not rearrange the raw download.
The split is fixed by `splits.json` / `case_split.csv` (shipped with the Zenodo outputs), so everybody uses exactly the
same cases.

Download *Task01_BrainTumour* from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (CC-BY-SA 4.0)
and sort its 484 labeled cases (`imagesTr` / `labelsTr`) into two folders using `case_split.csv`:

* cases with `split == test` go to `test/` (97 cases),
* all other cases (`train` and `val`, 339 + 48 = 387 cases) go to `train/`.

```
data/BrainTumour_MRI/                  (or any folder: set paths.data_root in configs/config.yaml)
├── train/
│   ├── images/BRATS_001.nii.gz ...    4-channel volumes, FLAIR is channel 0 (as in the MSD files)
│   └── labels/BRATS_001.nii.gz ...    same file name as the image; any label > 0 is tumor (merged to whole tumor)
└── test/
    ├── images/BRATS_002.nii.gz ...
    └── labels/BRATS_002.nii.gz ...
```

File names must be identical in `images/` and `labels/` and equal to the `case_id` of `case_split.csv`. The test list of
`splits.json` is checked against `test/images/` before inference, so a wrongly sorted folder is reported immediately.

Everything after that is automatic: `simulated-mri-artifact-stress-test prepare` builds the preprocessed training cache
(brain-mask crop, z-score inside the mask) and, if `splits.json` is not present, regenerates the same train / validation
split (seed 0, stratified by tumor-volume quartile) and verifies its fingerprint against the published one
(`data.split.expected_sha256_prefix`).

## 5. Run

The main pipeline in one go (`prepare`, `train`, `calibrate`, `infer`, `analyze`):

```bash
simulated-mri-artifact-stress-test run-all
simulated-mri-artifact-stress-test run-all --stages calibrate infer analyze    # only some stages, in order
```

The reader study is not part of `run-all`; run it separately after `calibrate`.

Stage by stage, for any model and any seed:

```bash
simulated-mri-artifact-stress-test prepare                                   # split + preprocessed training cache
simulated-mri-artifact-stress-test train --models unetr swinunetr --seeds 0 2
simulated-mri-artifact-stress-test calibrate                                 # SSIM-matched artifact strengths
simulated-mri-artifact-stress-test infer --models dynunet --seeds 1          # all 31 conditions for the selected checkpoints
simulated-mri-artifact-stress-test analyze                                   # needs inference of all configured models and seeds
simulated-mri-artifact-stress-test reader-study                              # volumes + blinded copy + slice overview for the radiologist
simulated-mri-artifact-stress-test status                                    # which runs are trained / inferred
```

Common options (every command): `-c my_config.yaml` and `--set key=value` (repeatable), for example
`--set training.epochs=2 --set paths.output_root=/data/run1`.

* **Long jobs (Kaggle, clusters).** Set `training.time_limit_hours` and `inference.time_limit_hours`
  (for example 11). The job stops cleanly before the limit; run the **same command** again to continue.
* **One checkpoint only.** `train --models unetr --seeds 1` followed by `infer --models unetr --seeds 1`.
* **Several GPUs.** Used automatically (`training.data_parallel: auto`).
* **Smoke test.** `infer --debug --set inference.debug_max_cases=1` writes to
  `simulated-mri-artifact-stress-test_Output/debug/` and never touches the real results.
* **Built-in sanity check.** Before any test prediction is stored, inference must reproduce the validation Dice
  recorded at the end of training (`inference.sanity_tolerance`), so a broken environment cannot silently produce
  wrong results.

## 6. Configuration (`configs/config.yaml`)

Nothing is hard-coded in the code: every setting of the paper runs lives in this one file.

| Section | What it controls |
|---|---|
| `paths` | input data folder, output folder, temporary cache |
| `experiment` | models and seeds that every stage processes by default |
| `data` | FLAIR channel, split seed, validation size, split fingerprint |
| `models` | architecture arguments and batch size of each model |
| `training` | patch size, schedule, learning rate, loss weights, augmentation, mixed precision, time limit |
| `artifacts` | levels, target SSIM step, strength caps, motion/noise/bias settings, frozen L1 to L5 strengths |
| `inference` | sliding window, threshold, NSD tolerance, saving masks and degraded volumes, reproduction tolerance |
| `analysis` | random seeds and bootstrap sizes of every analysis, level cap per artifact |
| `reader_study` | number of radiologist cases, levels, blinding |

## 7. Outputs

Everything the pipeline writes goes to `paths.output_root` (default `simulated-mri-artifact-stress-test_Output/`,
shipped on Zenodo):

```
simulated-mri-artifact-stress-test_Output/
├── splits.json                          train / val / probe / test case lists
├── case_split.csv                       case_id, split (train | val | test), probe flag
├── training/<model>/seed<k>/
│   ├── best.pt                          selected checkpoint (highest validation DSC)
│   ├── config.json  summary.json        hyper-parameters, best epoch, validation metrics, training time
│   └── train_log.csv  val_log.csv  val_cases.csv     per-epoch losses, validation metrics, per-case validation scores
├── calibration/
│   ├── degradation_config.json          the frozen artifact strengths of L1 to L10 (and which levels are capped)
│   ├── sweep_val.csv                    SSIM of every strength on every validation case
│   └── footprint_*.csv                  share of artifact energy in the low / mid / high frequency band
├── inference/
│   ├── predictions/<model>/<condition>/seed<k>/<case>.nii.gz     binary mask of every prediction, native space
│   │                                    condition = clean_L0 | motion_L1 ... motion_L10 | noise_L1 ... | bias_L1 ...
│   └── results/
│       ├── <model>/seed<k>.csv          one row per case and condition (columns below)
│       ├── results_test_long.csv        all 27,063 predictions in one table
│       ├── ssim_test.csv                SSIM achieved on every test case, artifact and level
│       ├── val_reproduction.csv         validation Dice reproduced by the inference code, per checkpoint
│       └── model_complexity.csv         parameters and GFLOPs
├── analysis/
│   ├── tables/                          statistics: model_summary, h1, h2, curves, envelope, volumetry, variance, tail, ...
│   ├── post_masks/                      mask-based analyses: error type, fragmentation, agreement, ensembles, hidden change
│   └── checkpoint_summary.csv  empty_predictions.csv  ssim_levels_test.csv  run_summary.json
├── reader_study/
│   ├── <case>/clean.nii.gz, <artifact>/L3|L6|L9.nii.gz      10 test cases shown to the radiologist
│   ├── blinded_key.csv  blinded_answers_template.csv    random id of every shown volume and the answer sheet for the reader
│   ├── radiology_slices.pdf             one page per case: original and every artifact at three levels
│   └── metadata.csv  selection.json     parameters and achieved SSIM of every shown volume
└── logs/                                logs of the runs
```

The probe flag marks a fixed subset of 16 validation cases that is recorded with the split for traceability; no stage
of this code depends on it.

Columns of `results/<model>/seed<k>.csv`:
`case_id, artifact, level, param, capped, ssim, dsc, iou, precision, recall, hd95, nsd, gt_ml, pred_ml, empty_pred`.

* `artifact = clean, level = 0` is the prediction on the unmodified scan.
* `param` is the artifact strength; `capped = 1` marks a level whose target SSIM was not reachable within the strength
  cap (motion and noise at L10, which therefore equal L9 in the analysis).
* `gt_ml` and `pred_ml` are the ground-truth and predicted tumor volumes in millilitres.

Only binary masks of the predictions are archived, not the degraded images: the degradation is deterministic
(one random pattern per case and artifact) and can be regenerated with `--set inference.save_degraded_volumes=true`.

## 8. Reuse the Zenodo outputs

```bash
# extract the downloaded zip so that simulated-mri-artifact-stress-test_Output/ is in the repository root
# (or point paths.output_root to its location)
simulated-mri-artifact-stress-test analyze        # recompute all statistics from the stored predictions (no GPU)
```

To rerun the inference from the released checkpoints (GPU), place the input data as in [section 4](#4-input-data) and
run `simulated-mri-artifact-stress-test infer`; to retrain from scratch, run `simulated-mri-artifact-stress-test train`.

## 9. Reproducibility notes

* Seeds, split, artifact patterns and bootstrap streams are fixed in the config; results are reproducible up to GPU
  floating-point differences in the predictions.
* The artifact strengths of the primary levels L1 to L5 are frozen in the config (`artifacts.frozen_primary`);
  `calibrate` checks that a fresh calibration reproduces them.
* The artifacts are applied on the native FLAIR volume and the result is standardized with the mask of the **clean**
  volume, so models see the same anatomy and normalization as on clean data.
* Models are trained **without** noise, blur, bias-field or motion augmentation, on purpose: the study measures the
  robustness of standard training, not of artifact-aware training.

## 10. Citation

The paper citation will be added here on publication. Until then, if you use this code or the released outputs,
please cite the Zenodo record:

> Banakhede, S. (2026). *Simulated MRI Artifact Stress Test: model checkpoints, predictions and analysis outputs*
> (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23645383

```bibtex
@dataset{banakhede_2026_zenodo_23645383,
  author    = {Banakhede, Sarang},
  title     = {Simulated MRI Artifact Stress Test: model checkpoints, predictions and analysis outputs},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23645383},
  url       = {https://doi.org/10.5281/zenodo.23645383}
}
```

GitHub also shows this citation through the **"Cite this repository"** button (from `CITATION.cff`).

If you use the data, please also cite the Medical Segmentation Decathlon:

> Antonelli, M., Reinke, A., Bakas, S. et al. The Medical Segmentation Decathlon. *Nature Communications* 13, 4128
> (2022). https://doi.org/10.1038/s41467-022-30695-9

## 11. License

* **Code** in this repository: [MIT License](LICENSE).
* **Released outputs** (checkpoints, predictions, tables) on Zenodo: under the license stated on the
  [Zenodo record](https://doi.org/10.5281/zenodo.23645383).
* **Input data**: the Medical Segmentation Decathlon images and labels are distributed under
  [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and are not included here; their terms apply to any
  use of the data.
