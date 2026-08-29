# Methodology

This document describes the methodology **as implemented** in the
supplied notebook (`notebooks/model.ipynb`, now modularized under
`src/`), and separately lists recommended future improvements. The two
are kept clearly distinct, per the project's documentation
requirements — nothing below is a claim about work that was not
actually done.

## 1. Dataset

The working corpus was built from two sources:

- A **normal-speech** collection: recordings from multiple speakers,
  each with an original `speaker_id`.
- A **stuttering-speech** collection: individual clips originally
  categorized, via filename, as either `fluent` or `dysfluent` speech
  from speakers who stutter.

From the notebook's own printed output at the time it was run:

| Category  | Recordings |
|-----------|-----------:|
| Normal    | 8,823      |
| Fluent    | 2,221      |
| Dysfluent | 2,490      |
| **Total** | **13,534** (13,535 before 1 file failed preprocessing) |

**Dataset source URL: Not provided in the notebook.** No download
link, dataset name/citation, or license statement was present in any
cell (source or markdown) of the supplied notebook. This is flagged
explicitly rather than invented — see `README.md` → "Dataset" and
`SECURITY_AUDIT.md`.

## 2. Label Construction (Implemented)

The three original recording-level categories were consolidated into a
binary classification target:

```python
def create_binary_label(category):
    if category in ["normal", "fluent"]:
        return 0
    if category == "dysfluent":
        return 1
    return -1
```

i.e. **Normal, Fluent → non_stuttered (0)**, **Dysfluent → stuttered
(1)**. This logic is implemented unchanged in
`src/data/load_dataset.create_binary_label()`. No recordings had an
unresolved ("unknown") category after filename-based category
assignment, per the notebook's own printed check.

## 3. Audio Preprocessing (Implemented)

Implemented in `src/data/preprocess.py`, unchanged from the notebook:

1. Load and resample to 16 kHz, mono.
2. Replace non-finite (NaN/Inf) samples with zero.
3. Trim leading/trailing silence (`librosa.effects.trim`, `top_db=30`).
4. Discard recordings shorter than 0.20 s after trimming.
5. Peak-amplitude normalize (`x / max(|x|)`).
6. Persist the cleaned waveform as a new 16 kHz mono WAV file.

Of 13,535 original files, 13,534 were processed successfully; 1 failed
(logged with a reason code in `failed_files.csv`).

## 4. Speaker Identification (Implemented)

- Normal-speech recordings retain their original `speaker_id`.
- Stuttering-dataset recordings have their speaker code parsed from a
  filename prefix matching `^([MF]_\d+)_` (e.g.
  `M_0030_16y4m_1_dysfluent_000.wav` → `M_0030`). Filenames that do not
  match this pattern keep their original `speaker_id` as a fallback.

Implemented in `src/data/load_dataset.fix_speaker_id()`.

## 5. Speaker-Independent Dataset Splitting (Implemented)

Partitioning is performed **at the speaker level**, independently
within the normal-speaker group and the stuttering-speaker group,
using a 70%/15%/15% split (`sklearn.model_selection.train_test_split`,
`random_state=42`) for train/validation/test respectively, before
recombining both groups' assigned speakers into the final speaker
sets. All recordings belonging to a given speaker are assigned as a
block to that speaker's partition.

**Data leakage check.** The implementation explicitly verifies, after
splitting, that `Train ∩ Validation = Train ∩ Test = Validation ∩ Test
= ∅` at the speaker-ID level, and that both classes are present in
every partition. This check is preserved in
`scripts/prepare_dataset.split_speakers_independently()`, which raises
an error if leakage is ever detected. **No speaker leakage was found**
in the original notebook's implementation or its printed verification
output — the split, as written, is genuinely speaker-independent. No
"legacy/leaky" variant was created because none existed in the source
material.

## 6. Feature Extraction (Implemented)

Implemented in `src/features/feature_extraction.py`, unchanged:

- Each cleaned waveform is silence-trimmed again, then
  truncated/zero-padded to a fixed 5.0 s window (80,000 samples @
  16 kHz).
- Peak-amplitude normalization is applied a second time at this stage.
- 40 MFCCs are computed (`n_mfcc=40`, `n_fft=512`, `hop_length=256`).
- First-order (delta) and second-order (delta-delta) time derivatives
  of the MFCCs are computed and stacked with the static coefficients:
  `features = vstack([mfcc, delta, delta2])` → 120 feature rows.
- Each of the 120 rows is standardized independently (zero mean, unit
  variance) across its own time axis, per recording.
- **Final feature tensor shape: (120, 313)**, reshaped to
  `(120, 313, 1)` before being passed to the CNN.

| Property | Value |
|---|---|
| Feature type | MFCC + Δ (delta) + ΔΔ (delta-delta) |
| MFCC coefficients | 40 |
| FFT size | 512 |
| Hop length | 256 |
| Fixed audio window | 5.0 s (80,000 samples @ 16 kHz) |
| Input tensor shape | (120, 313, 1) |
| Normalization | Per-recording, per-coefficient z-score |

## 7. Model Architectures (Implemented)

The notebook contains **two** CNN architectures, trained sequentially
and preserved as-is in `src/models/model.py`:

### CNN V1 (`build_cnn_v1`)

Three `Conv2D → BatchNorm → MaxPool2D → Dropout` blocks
(32 / 64 / 128 filters, 3×3 kernels, dropout 0.25 / 0.30 / 0.35),
`GlobalAveragePooling2D`, a 128-unit dense + BatchNorm + 0.50 dropout
layer, and a sigmoid output. Checkpoint: `best_cnn_model.keras`.

### CNN V2 (`build_cnn_v2`)

Three `Conv2D → BatchNorm → MaxPool2D → SpatialDropout2D` blocks
(16 / 32 / 64 filters, 3×3 kernels, L2 kernel regularization `1e-4`,
spatial dropout 0.20 / 0.25 / 0.30), `GlobalAveragePooling2D`, a
64-unit L2-regularized dense + 0.50 dropout layer, and a sigmoid
output. Checkpoint: `cnn_v2_best.keras`.

**CNN V2 is the architecture referenced throughout the accompanying
research paper** and is the source of the project's reported results
(Section 8 below). Both architectures are preserved unmodified; the
project's scripts default to training/evaluating CNN V2, with CNN V1
available via a `--model-version v1` flag.

## 8. Training (Implemented)

Implemented in `src/training/train.py`, unchanged:

| Parameter | CNN V1 | CNN V2 |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 0.001 | 0.0003 |
| Loss | binary_crossentropy | binary_crossentropy |
| Batch size | 32 | 64 |
| Epochs (max) | 40 | 25 |
| Class weighting | `sklearn.compute_class_weight("balanced")` | none |
| Early stopping | `val_loss`, patience 8 | `val_auc` (max), patience 5 |
| LR reduction | `val_loss`, factor 0.5, patience 3 | `val_auc` (max), factor 0.5, patience 2 |
| Checkpoint metric | `val_loss` (min) | `val_auc` (max) |
| Random seed | 42 (numpy + TensorFlow) | 42 (numpy + TensorFlow) |

## 9. Evaluation (Implemented)

Implemented in `src/evaluation/evaluate.py`, unchanged: probability =
`model.predict(X_test).ravel()`; predicted class = probability ≥ 0.5;
accuracy, precision, recall, F1 (positive class = "stuttered"), and
ROC-AUC (from the continuous probability) are computed via
scikit-learn, alongside a full classification report and confusion
matrix.

### Reported test-set results (CNN V2, held-out speaker-disjoint test set)

| Metric | Value |
|---|---:|
| Accuracy | 91.54% |
| Precision (Stuttered) | 71.77% |
| Recall (Stuttered) | 96.35% |
| F1-score (Stuttered) | 82.26% |
| ROC-AUC | 96.89% |

These figures are taken directly from the notebook's own printed
output (cell "FINAL TEST RESULTS" / `cnn_v2_metrics.csv`) and are
reproduced here, not recomputed or estimated.

### Note on speaker counts vs. the accompanying paper

Re-running the split logic exactly as implemented in the notebook
produces **30 train / 6 validation / 8 test speakers** (44 total: 28
normal-speaker + 16 stuttering-speaker), as printed directly in the
notebook's own "SPLIT COMPLETE" / "FINAL SPLIT VERIFICATION" cell
outputs. The accompanying paper's Table 3 instead reports **30 train /
7 validation / 7 test speakers**. This is a discrepancy between the
paper's reported table and the notebook's own printed verification
output; it is stated here explicitly rather than silently reconciled,
consistent with this project's "do not fabricate" documentation
policy. The recording-level totals (9,385 / 1,997 / 2,152) and the
zero-speaker-overlap leakage check match between the notebook output
and the paper in all cases checked.

## 10. Known Data-Leakage / Methodological Considerations

- **Speaker leakage:** checked and ruled out (Section 5).
- **Class imbalance:** the Stuttered class is a minority class overall
  (~18% of recordings). `class_weight="balanced"` is applied for CNN
  V1 but not CNN V2 in the original implementation — this asymmetry is
  preserved as-is rather than "fixed," per the instruction to document
  rather than silently change existing methodology.
- **Recording-condition confound:** the normal-speech and
  stuttering-speech recordings originate from different source
  collections; residual channel/recording-condition differences cannot
  be ruled out as a confound distinct from fluency itself. This is not
  addressed in the current implementation.

## 11. Reproducibility Notes

- Python version / hardware (CPU vs GPU) used for the original run were
  **not recorded** in the supplied notebook and are not asserted here.
- Random seed: 42 (dataset splitting, NumPy, TensorFlow).
- All feature-extraction and model hyperparameters are hard-coded
  exactly as found in the source notebook (see `config/config.py`).

## Recommended Future Improvements (NOT implemented)

These are explicitly **not** part of the current implementation and
are listed only as directions a future iteration of this project could
take:

- Cross-corpus evaluation against public stuttering benchmarks (e.g.
  SEP-28k, UCLASS, FluencyBank).
- Multi-class disfluency-type classification (repetition, prolongation,
  block, interjection) instead of the current binary target.
- Ablation studies (e.g. removing delta/delta-delta channels, varying
  MFCC count, comparing against log-Mel spectrogram input).
- Self-supervised acoustic representations (e.g. Wav2Vec2.0 embeddings)
  as an alternative or complement to MFCC-based features.
- Explainability analysis (e.g. saliency/attention over the feature
  tensor) to identify which cepstral-temporal regions drive
  predictions.
- Expanding the speaker pool beyond the current scale to reduce
  variance in the held-out speaker-level test estimate.
- External clinical validation; no diagnostic or therapeutic claim is
  made or implied by the current implementation.
