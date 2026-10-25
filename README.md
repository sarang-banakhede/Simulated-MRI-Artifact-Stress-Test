# Simulated MRI Artifact Stress Test

Code of the paper **"What Clean Dice Hides: Severity-Matched Stress Testing of 3D Brain Tumor Segmentation under Simulated MRI Artifacts"**.

A mean Dice score on clean MRI says little about what happens to an individual patient when the scan is degraded. This
repository trains three 3D segmentation models, degrades the held-out test volumes with three physically motivated
artifacts at ten severities of **equal structural similarity (SSIM 0.95 to 0.50)**, and measures what the mean Dice
conceals: individual-level operating limits, boundary damage, tumor-volume change and the sources of variation.

| | |
|---|---|
| Data | Medical Segmentation Decathlon *Task01 BrainTumour* (BraTS), FLAIR only, whole tumor. 339 train / 48 validation / 97 test cases |
| Models | DynUNet, UNETR, SwinUNETR (MONAI), trained from scratch, 3 seeds each = 9 checkpoints |
| Artifacts | k-space **motion**, **Rician noise**, polynomial **bias field**; levels L1 to L10 (L1 to L5 primary, L6 to L10 exploratory) |
| Scale | 9 checkpoints x 97 test cases x 31 conditions = **27,063 predictions** |
| Metrics | DSC, HD95, normalized surface Dice (NSD), predicted tumor volume |

**Outputs (weights, predictions, results, analysis tables) are not stored in this repository.**
They are archived on Zenodo: **[zenodo.org/records/23645383](https://zenodo.org/records/23645383)**. Download the single zip (`simulated-mri-artifact-stress-test_Output.zip`) and extract it in the repository root, so that
the folder `simulated-mri-artifact-stress-test_Output/` exists, to rerun the analysis without
retraining (see [Reuse the Zenodo outputs](#8-reuse-the-zenodo-outputs)).

---

## 1. Pipeline

```
prepare ──> train ──> calibrate ──> infer ──> analyze
  │           │           │           │          │
  │           │           │           │          └─ statistics, operating limits, volumetry, variance,
  │           │           │           │             mask-based analysis (error types, agreement, ensembles)
  │           │           │           └─ 31 conditions x 97 test cases x every checkpoint
  │           │           └─ artifact strength that gives the target SSIM of each level (validation cases)
  │           └─ 3 models x 3 seeds, resumable, best checkpoint by validation DSC
  └─ patient-level split + preprocessed training cache
```

Each stage is a command of the `simulated-mri-artifact-stress-test` CLI, can be run alone, and is **safe to rerun**: finished work is
skipped, interrupted training and inference resume where they stopped.

## 2. Repository layout

```
configs/config.yaml          the only configuration file (paths, split, models, training, artifacts, inference, analysis)
src/simulated_mri_artifact_stress_test/
  cli.py                     command line: prepare | train | calibrate | infer | analyze | reader-study | run-all | status
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
    results.py               merge of all results, per-checkpoint summary
    statistics.py            clean-Dice vs robustness (H1), pairwise reversals (H2), curves, tail risk, size, ranks
    operating.py             operating envelopes, clinical volumetry (ICC, limits of agreement), variance decomposition
    post_masks.py            mask-based analysis: error type, fragmentation, seed/model agreement, ensembles
    sensitivity.py           SSIM-restricted and adjusted analyses, pass-share sensitivity, relative-loss variance
  reader_study.py            clean and degraded volumes, blinded copy and slice overview for a radiologist
tests/                       unit tests (fast) and one end-to-end test on a synthetic dataset
```

## 3. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                    # 20 fast unit tests, no data needed
```

Python 3.10 or newer. Training and inference need a CUDA GPU (a 4 GB GPU works with `inference.sw_batch_size: 1`).
Versions of the paper runs: PyTorch 2.11, MONAI 1.6.1, NumPy 2.1, SciPy 1.16, scikit-image 0.25.

## 4. Input data

The pipeline **expects the data already arranged** in the structure below; it does not rearrange the raw download.
The split is fixed by `splits.json` / `case_split.csv` (shipped with the Zenodo outputs), so everybody uses the same
cases. Download *Task01_BrainTumour* from the [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
(CC-BY-SA 4.0) and sort its 484 labeled cases (`imagesTr` / `labelsTr`) into two folders:

* cases with `split == test` in `case_split.csv` go to `test/` (97 cases),
* all other cases (`train` and `val`, 387 cases) go to `train/`.

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
Everything after that is automatic: `simulated-mri-artifact-stress-test prepare` builds the preprocessed training cache (brain-mask crop,
z-score inside the mask) and, if `splits.json` is not present, regenerates the same train / validation split
(seed 0, stratified by tumor-volume quartile) and verifies its fingerprint against the published one.

## 5. Run

Everything in one go:

```bash
simulated-mri-artifact-stress-test run-all
```

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
* **Smoke test.** `infer --debug --set inference.debug_max_cases=1` writes to `simulated-mri-artifact-stress-test_Output/debug/`.
* Before any test prediction is stored, inference must reproduce the validation Dice recorded at the end of
  training (`inference.sanity_tolerance`), so a broken environment cannot produce silent results.

## 6. Configuration (`configs/config.yaml`)

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

Everything the pipeline writes goes to `paths.output_root` (default `simulated-mri-artifact-stress-test_Output/`, shipped on Zenodo):

```
simulated-mri-artifact-stress-test_Output/
├── splits.json                          train / val / probe / test case lists
├── case_split.csv                       case_id, split (train | val | test), probe flag: which case was used where 
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

Columns of `results/<model>/seed<k>.csv`: `case_id, artifact, level, param, capped, ssim, dsc, iou, precision, recall,
hd95, nsd, gt_ml, pred_ml, empty_pred`. `artifact = clean, level = 0` is the prediction on the unmodified scan;
`param` is the artifact strength, `capped = 1` marks a level whose target SSIM was not reachable within the cap.

Only binary masks of the predictions are archived, not the degraded images: the degradation is deterministic
(one random pattern per case and artifact) and can be regenerated with `--set inference.save_degraded_volumes=true`.

## 8. Reuse the Zenodo outputs

```bash
# put the downloaded archive at simulated-mri-artifact-stress-test_Output/  (or set paths.output_root to its location)
simulated-mri-artifact-stress-test analyze        # recompute all statistics from the stored predictions (no GPU, a few minutes)
```

To rerun the inference from the released checkpoints (GPU), place the input data as in section 4 and run
`simulated-mri-artifact-stress-test infer`; to retrain, run `simulated-mri-artifact-stress-test train`.

## 9. Notes

* Seeds, split, artifact patterns and bootstrap streams are fixed in the config; results are reproducible up to GPU
  floating-point differences in the predictions.
* The artifacts are applied on the native FLAIR volume and the result is standardized with the mask of the **clean**
  volume, so models see the same anatomy and normalization as on clean data. Models are trained without noise, blur,
  bias-field or motion augmentation on purpose.
* Citation and license information will be added with the public release of the paper.
