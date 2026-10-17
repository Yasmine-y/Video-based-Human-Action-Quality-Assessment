# Video-based Human Action Quality Assessment: AQA-7 Diving

This repository predicts the judges' score of an Olympic dive from its video, using frozen pretrained I3D features and small regression models. It compares a global (time-averaged) baseline with four temporal models, then runs one controlled experiment on whether temporal modelling helps (Task 3, Option A).

## Contents

| File | What it is |
|---|---|
| `aqa_diving.ipynb` | The whole pipeline in one notebook, saved **with its outputs** so it can be read without running it. Parts: 1 data exploration and split, 2 I3D feature extraction, 3 baseline and temporal models, 4 Task 3 experiment. |
| `requirements.txt` | Python packages needed to run the notebook locally. |
| `report.pdf` | The report (≤ 5 pages) and the research proposal. |

## Dataset

AQA-7 (Parmar & Morris, *Action Quality Assessment Across Multiple Actions*, WACV 2019). Only **10 m platform diving** (action class 1, 370 clips) and the official **Split-4** train/test lists are used.

Expected structure (`ROOT` must point to this `AQA-7/` folder). The other sports folders in `Actions/` can be present; they are not used.

```
AQA-7/
├── Actions/
│   └── diving/
│       ├── 001.avi
│       ├── 002.avi
│       └── ...            (370 videos)
├── Split_4/
│   ├── split_4_train_list.mat
│   └── split_4_test_list.mat
└── Readme.txt
```

## How to run

There are two options. In both, **set `ROOT` to your own path to the AQA-7 folder** (the folder that contains `Actions/` and `Split_4/`). It is in the first code cell of Part 1.

### Option A: Kaggle (no download needed)

1. Create a Kaggle notebook and import `aqa_diving.ipynb` (File → Import Notebook).
2. Attach the dataset copy already on Kaggle: <https://www.kaggle.com/datasets/yasminecharradi/aqa-7-sportssss> (Add Input → search for it).
3. Settings: **Accelerator: GPU** (the notebook uses one GPU; it was run on a T4) and **Internet: on** (Part 2 downloads the I3D model).
4. In the first code cell of Part 1, set `ROOT` to where Kaggle put the dataset (find it in the *Input* panel on the right, under `/kaggle/input/`). Then choose **Run All**.

Results are written to `/kaggle/working/`.

### Option B: on your own machine

1. Download and unzip the dataset:
   ```bash
   mkdir AQA-Seven && cd AQA-Seven
   wget http://rtis.oit.unlv.edu/datasets/AQA-7.zip
   unzip AQA-7.zip
   cd ..
   ```
   This download link comes from another public repository that uses AQA-7. It could not be tested while preparing this submission. If it does not work, use the Kaggle copy above (it can also be downloaded from that page).
2. Install the packages (the reported run used Python 3.13; a CUDA GPU is strongly recommended). `requirements.txt` pins the versions listed under *Compute resources and library versions*:
   ```bash
   pip install -r requirements.txt
   ```
3. In the first code cell of Part 1, set `ROOT` to the unzipped AQA-7 folder (for example `"AQA-Seven/AQA-7"`) and `OUT` to a folder for the results (for example `"outputs"`).
4. Open the notebook in Jupyter and run all cells. Internet access is needed once, to download the I3D model.

## What the notebook saves, and re-running

All outputs go to the folder `OUT` (`/kaggle/working/` on Kaggle):

- `diving_splits.csv` is the train/val/test split (240 / 60 / 70). `diving_splits_meta.json` records how it was made.
- `features/` holds the I3D features (370 × 9 × 1024) and the video number of each row.
- `i3d/` holds the downloaded I3D code and weights.
- `results/` holds the CSV tables, the chosen settings (`best_configs.json`), the test predictions of every model and seed, and the figures.

The slow steps (feature extraction, hyperparameter selection, final training) are **skipped when their files already exist**, so a second run only reloads them. To recompute a step, delete its file or folder: `features/`, `results/selection.csv`, or `results/final_runs.csv` with `results/test_predictions.csv`. On Kaggle, `/kaggle/working/` is emptied when the session ends. Use *Save Version* (or download the folder) to keep it.

## Compute resources and library versions

The reported results were produced on **Kaggle** with **one NVIDIA Tesla T4 GPU** (about 15 GB of GPU memory). The Kaggle session offers two T4s, but the notebook uses only one. A GPU is strongly recommended: Part 2 runs I3D on all 370 clips, and Part 3 trains 145 small models (120 for hyperparameter selection, 25 final runs). For reference, the quick test in section 3.4 (one small GRU run) took 5.5 s on the T4; larger settings take longer. Total run times were not recorded. The features and models are small, so memory is not a constraint.

Library versions of the reported run, as printed by the notebook:

| Library | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| NumPy | 2.1.3 |
| pandas | 2.3.3 |
| SciPy | 1.16.3 |
| scikit-learn | 1.6.1 |
| OpenCV | 4.14.0 |
| Matplotlib | Kaggle default (not printed) |

## Reproducibility

- **Seeds:** split seed 42; selection seeds 0, 1, 2; final seeds 100–104; bootstrap seed 0. Deterministic cuDNN/cuBLAS settings are switched on. Small differences can still appear on different GPUs or library versions.
- **Pretrained model:** I3D, PyTorch port by piergiaj (<https://github.com/piergiaj/pytorch-i3d>), weights `rgb_imagenet.pt` (ImageNet + Kinetics-400). The notebook prints the SHA-256 of the downloaded files. The run reported here used:
  - `pytorch_i3d.py`: `c322c3ca015dab5ee43e4d15707149d510b98931406c4a5d58b27934ed6777ed`
  - `rgb_imagenet.pt`: `d64d773a81767db7af4f102b46a221db2251fa4e2fff37d9039016033161a247`
- **Protocol:** the test set is used only once, for the final runs. Every choice (validation split, hyperparameters, early stopping, the main Task 3 comparison) uses the validation set.

## Main results (test set, 70 dives)

| Model | Spearman ρ (average of 5 seeds) [95 % CI] | MAE in points [95 % CI] |
|---|---|---|
| avgpool (baseline) | 0.776 [0.638, 0.868] | 7.31 [5.96, 8.83] |
| attn (temporal, main comparison) | 0.766 [0.621, 0.861] | 6.24 [5.16, 7.40] |
| ridge regression | 0.755 [0.619, 0.852] | 9.28 [7.19, 11.46] |

On the same test dives, attention pooling reduced MAE by 1.07 points compared with average pooling (95 % CI [−2.00, −0.23]). It did not improve ranking (Δρ −0.010, CI [−0.060, 0.033]). All models and the detailed comparisons are in the notebook (sections 3.9 and 4.3).

## Known limitations

- The dataset is small (240 training clips, 70 test clips), so confidence intervals are wide.
- Only final scores are available, so dive difficulty and execution quality cannot be separated.
- All clips appear to come from one competition, so the same athletes are probably in train and test.
- The I3D features are frozen, and time is coarse (9 snippets of about 0.5 s).
- The Task 3 experiment does not show *why* attention helps: no shuffled-order control was run.
- The validation set is small (60 clips), so model selection is noisy: validation ρ (≈ 0.6) is lower than test ρ (≈ 0.75) for every model.
- Results can vary slightly on other GPUs or library versions, even with fixed seeds.
- The direct download link for AQA-7 (Option B) could not be tested; the Kaggle copy is the reliable source.
