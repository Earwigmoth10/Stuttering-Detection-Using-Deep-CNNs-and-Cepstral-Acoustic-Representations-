# Stuttering Detection from Speech Using Machine Learning

A speaker-independent binary classifier that distinguishes stuttered
from non-stuttered speech recordings using a convolutional neural
network (CNN) operating on MFCC-based cepstral features.

This repository is a cleaned, modular, and secure refactor of an
original exploratory Jupyter notebook. The research logic (dataset
labels, preprocessing, feature extraction, train/test methodology,
model architectures, and evaluation) is preserved exactly as
implemented — see `SECURITY_AUDIT.md` and `docs/methodology.md` for a
full account of what was changed (paths/security only) and what was
not (all research logic).

## Overview

The pipeline takes raw speech recordings from two sources — a
normal-speech collection and a stuttering-speech collection (whose
clips are pre-labeled by filename as `fluent` or `dysfluent`) — and
trains a CNN to classify each recording as **Non-stuttered** or
**Stuttered**. Partitioning into train/validation/test sets is
performed at the **speaker level**, so no speaker's recordings appear
in more than one split, and this property is verified
programmatically.

## Research Objective

Classify a short speech recording as containing stuttered
("dysfluent") speech or not, using only acoustic (MFCC-based) features
and a CNN, under a speaker-independent evaluation protocol.

## Dataset

- **Sources:** a normal-speech recording collection (multiple
  speakers) and a stuttering-speech collection (clips per speaker,
  labeled `fluent` or `dysfluent` via filename).
- **Original three-way categories → binary target:**
  `Normal, Fluent → non_stuttered (0)`; `Dysfluent → stuttered (1)`.
- **Dataset source URL: Not provided in the notebook.** No dataset
  name, citation, download link, or license was present in the
  supplied source material. If you know this project's dataset
  provenance, please add it here and to `docs/methodology.md`.
- **Dataset license:** Not specified in the provided notebook. Because
  of this, the dataset itself is **not redistributed** in this
  repository (see `data/README.md`).

## Dataset Statistics

(As printed directly by the original notebook; see
`docs/methodology.md` for full detail and the exact split table.)

| Category | Recordings |
|---|---:|
| Normal | 8,823 |
| Fluent | 2,221 |
| Dysfluent | 2,490 |
| **Total (after preprocessing)** | **13,534** (of 13,535 originally; 1 file failed — too short after silence trimming) |

Speaker-independent split (as produced by
`scripts/prepare_dataset.py`, matching the notebook's own printed
output): **30 train / 6 validation / 8 test speakers** (44 total: 28
normal-speaker, 16 stuttering-speaker). See
`docs/methodology.md` for a note on a minor discrepancy between this
and the accompanying paper's reported validation/test speaker counts.

| Split | Recordings | Speakers | Non-stuttered | Stuttered |
|---|---:|---:|---:|---:|
| Train | 9,385 | 30 | 7,942 | 1,443 |
| Validation | 1,997 | 6 | 1,388 | 609 |
| Test | 2,152 | 8 | 1,714 | 438 |

## Methodology

1. **Dataset loading** — read `audio_dataset.csv` metadata; separate
   normal vs. stuttering-source recordings.
2. **Preprocessing** — resample to 16 kHz mono, remove NaN/Inf samples,
   trim silence (`top_db=30`), discard recordings < 0.20 s, peak-amplitude
   normalize, persist cleaned audio.
3. **Label processing** — derive the original three-way category from
   source/filename, then collapse to the binary target
   (`normal, fluent → 0`; `dysfluent → 1`); recover per-recording
   speaker IDs.
4. **Feature extraction** — fixed 5.0 s window; 40 MFCCs (`n_fft=512`,
   `hop_length=256`) plus delta and delta-delta derivatives, stacked to
   a (120, 313) tensor, standardized per recording/coefficient.
5. **Dataset splitting** — speaker-independent 70/15/15 split
   (separately within normal- and stuttering-speaker groups, then
   recombined), with a programmatic zero-overlap leakage check.
6. **Model architecture** — two CNN variants are implemented and
   preserved (`src/models/model.py`): **CNN V1** (32/64/128 filters) and
   **CNN V2** (16/32/64 filters, L2-regularized, SpatialDropout). CNN
   V2 is the architecture referenced in the accompanying paper and the
   source of the reported results below.
7. **Training** — Adam optimizer, binary cross-entropy loss, early
   stopping, learning-rate reduction on plateau, and best-checkpoint
   saving (see `docs/methodology.md` for exact per-version hyperparameters).
8. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion
   matrix, and classification report on the held-out speaker-disjoint
   test set.

## Reported Results (CNN V2, held-out test set)

| Metric | Value |
|---|---:|
| Accuracy | 91.54% |
| Precision (Stuttered) | 71.77% |
| Recall (Stuttered) | 96.35% |
| F1-score (Stuttered) | 82.26% |
| ROC-AUC | 96.89% |

These are the values printed by the original notebook / saved to
`cnn_v2_metrics.csv`; they are not recomputed, estimated, or
fabricated here.

## Project Structure

```text
stuttering-detection/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── config/
│   └── config.py
├── data/
│   ├── raw/                 (git-ignored; place your dataset here)
│   ├── processed/           (git-ignored; generated)
│   └── README.md
├── notebooks/
│   └── model.ipynb          (cleaned research/demo notebook)
├── src/
│   ├── data/
│   │   ├── load_dataset.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── feature_extraction.py
│   ├── models/
│   │   └── model.py
│   ├── training/
│   │   └── train.py
│   └── evaluation/
│       └── evaluate.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── full_pipeline.py
├── results/
│   ├── figures/
│   └── metrics/
├── docs/
│   └── methodology.md
└── SECURITY_AUDIT.md
```

## Installation

```bash
git clone <this-repository-url>
cd stuttering-detection
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then edit .env with your dataset path
```

## Usage

Configure your dataset location first — either set the
`DATASET_ROOT` environment variable, edit `.env` (copied from
`.env.example`), or place your dataset under `data/raw/`. See
`data/README.md` for the expected structure.

Run the stages individually:

```bash
python scripts/prepare_dataset.py
python scripts/train_model.py --model-version v2
python scripts/evaluate_model.py --model-path models/cnn_v2_best.keras --prefix cnn_v2
```

...or run the entire pipeline in one command:

```bash
python scripts/full_pipeline.py --model-version v2
```

The cleaned, narrated version of the original workflow is also
available as a notebook for exploratory use:

```bash
jupyter notebook notebooks/model.ipynb
```

## Reproducibility

- **Python:** 3.10+ recommended (exact version used for the original
  run was not recorded in the source notebook).
- **Dependencies:** pinned ranges in `requirements.txt`.
- **Random seed:** 42 (dataset splitting, NumPy, TensorFlow) —
  configured in `config/config.py`.
- **Dataset:** must be obtained separately; see `data/README.md`.
- **Configuration:** all paths and hyperparameters are centralized in
  `config/config.py` and `.env` (see `.env.example`).

## Security and Privacy

- No machine-specific local paths are present anywhere in this
  repository's source code; all paths are resolved via
  `config/config.py` from environment variables / a `.env` file, with
  a project-relative fallback.
- No credentials, API keys, tokens, or passwords were found in the
  original notebook, and none are present here.
- The audio dataset is **not** committed to this repository (raw audio,
  processed audio, and extracted feature arrays are all git-ignored).
- `.env` (with your real dataset path) is git-ignored; only
  `.env.example` (placeholders) is committed.
- See `SECURITY_AUDIT.md` for the complete audit of issues found in the
  original notebook and the fixes applied.

## Limitations

(See `docs/methodology.md` §10 for full detail.)

- Evaluation relies on a modest speaker pool (44 speakers total; 6–8
  held out for validation/test), so performance estimates carry
  non-trivial variance.
- The binary label collapses `Fluent` and `Normal` into one class,
  discarding finer-grained disfluency-type information.
- Normal-speech and stuttering-speech recordings come from different
  source collections; residual recording-condition differences cannot
  be ruled out as a confound.
- No cross-corpus or clinical validation was performed. This system
  makes no diagnostic or therapeutic claim.
- Python/library versions and training hardware used for the original
  run were not recorded in the source material.
