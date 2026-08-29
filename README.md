# Speaker-Independent Stuttering Detection Using CNNs and Cepstral Acoustic Representations

A convolutional neural network (CNN) system that classifies speech recordings as **Non-stuttered** or **Stuttered**, trained and evaluated under a strict **speaker-independent** protocol (no speaker's voice appears in more than one of the train/validation/test splits).

> **Note on scope:** the working system implemented and evaluated in this repository (`model.ipynb`, and the accompanying research paper) performs **binary** classification — **Non-stuttered vs. Stuttered**. The original data is drawn from three source-level recording categories (**Normal**, **Fluent**, **Dysfluent**), which are consolidated into the binary target as described in [Dataset Label Transformation](#dataset-label-transformation) below.

`Python` · `TensorFlow / Keras` · `librosa` · `scikit-learn`

---

##  Overview

Stuttering is a speech-fluency disorder characterized by disruptions in the timing, rhythm, and smooth production of speech — sound/syllable repetitions, prolongations, and blocks. Clinical assessment has traditionally relied on manual perceptual judgment by a speech-language pathologist, which is time-consuming and susceptible to inter-rater disagreement. This motivates automatic, signal-processing- and deep-learning-based approaches that can support fluency screening, therapy monitoring, and longitudinal tracking.

<img width="1536" height="1024" alt="ChatGPT Image Aug 11, 2026, 11_01_56 PM" src="https://github.com/user-attachments/assets/3eb1445c-9ef9-48ba-a5e9-25f2c758c28b" />
)
*Types of speech fluency disorders — normal fluent speech, developmental stuttering (repetition, prolongation, block), and other fluency disorders, together with common manifestations and documented impacts of stuttering.*

This project proposes a CNN-based binary stuttering detector (**Non-stuttered** vs. **Stuttered**) trained on 16 kHz audio represented as stacked **MFCC + Δ + ΔΔ (delta / delta-delta)** cepstral feature tensors, under an explicit, programmatically-verified **speaker-disjoint** train/validation/test split.

Key properties of the system:
- **Speaker-independent evaluation** — speaker identities are recovered from filenames/metadata, grouped, and split so that no speaker seen in training appears in validation or testing. This is checked programmatically (pairwise intersection of speaker-ID sets).
- **Cepstral (MFCC-based) acoustic representation** — 40 MFCCs plus their first- and second-order temporal derivatives, stacked into a 120 × 313 feature tensor, distinct from the waveform/spectrogram images used only for illustrative visualization in this repo's figures.
- **Class-imbalance-aware evaluation** — precision, recall, F1, and ROC-AUC are reported alongside accuracy because the Stuttered class is a minority class (~18% of the corpus).

---

##  Objectives

- Build a complete, reproducible **preprocessing → feature-extraction → CNN classification** pipeline for binary stuttering detection, with parameters (sample rate, window/hop size, feature dimensionality) traceable to the implementation.
- Apply and **verify** a genuinely speaker-independent partitioning strategy at the level of individual speakers (not individual recordings), with an explicit, programmatic leakage check.
- Represent speech acoustically using **MFCC + Δ + ΔΔ** cepstral features (rather than, or in addition to, spectrogram/log-Mel input), and evaluate the trained CNN (**CNN V2**) on a held-out, speaker-disjoint test set.
- Report test performance (accuracy, precision, recall, F1, ROC-AUC) and numerically re-verify it against the raw confusion matrix.

---

##  Proposed Methodology

```
Audio Input
  → Preprocessing (resample 16 kHz mono, silence trim, duration filter, peak normalization)
  → Binary Label Transformation (Normal/Fluent → Non-stuttered, Dysfluent → Stuttered)
  → Speaker Identification & Speaker-Independent Split (70% / 15% / 15%)
  → Feature Extraction (40 MFCC + Δ + ΔΔ → 120 × 313 tensor)
  → CNN V2 (3× Conv2D/BN/Pool/Dropout blocks → GAP → Dense → Sigmoid)
  → Classification (0.5 threshold) → Non-stuttered / Stuttered
```

<img width="1608" height="978" alt="ChatGPT Image Aug 11, 2026, 11_37_44 PM" src="https://github.com/user-attachments/assets/c1194737-df18-4cc9-9ab0-ba73dfd9ff4d" />

*End-to-end framework: raw audio dataset → data cleaning → audio preprocessing → MFCC-based feature extraction → CNN V2 classification → prediction output, plus the speaker-independent train/validation/test partitioning used for evaluation.*

The pipeline comprises seven sequential stages, matching the implementation: (1) deterministic audio preprocessing, (2) binary label transformation of the original three-way labels, (3) speaker identification and speaker-independent partitioning, (4) cepstral feature extraction (MFCC + Δ + ΔΔ), (5) CNN feature learning (CNN V2), (6) class-weighted / regularized optimization with adaptive learning-rate control and early stopping, and (7) evaluation on a held-out, speaker-disjoint test partition.

---

##  Dataset

The working corpus was built from **two sources**: a normal-speech recordings collection, and a stuttering-speech dataset whose clips were originally categorized (by filename) as **Fluent** or **Dysfluent** speech from speakers who stutter.

### Dataset 1 — Stuttering speech (UCLASS-derived)

- **Classes used:** Fluent, Dysfluent (speaker-code filename prefix, e.g. `M_0030_16y4m_1_dysfluent_000.wav` → speaker `M_0030`)
- **Purpose:** source of the **Stuttered** class (Dysfluent clips) and part of the **Non-stuttered** class (Fluent clips)
- **Likely underlying archive:** [UCLASS — University College London Archive of Stuttered Speech](https://www.uclass.psychol.ucl.ac.uk/) (Howell, Davis & Bartrip, 2009), based on the UCLASS-style filename convention and dysfluent-clip structure present in this project's data.
- **Candidate Kaggle mirror (not independently confirmed against this project's exact files — verify before citing):** [UCLASS Stuttered Speech Clips (SEP-28k Format)](https://www.kaggle.com/datasets/vudominhgiang/uclass-stuttered-speech-clips-sep-28k-format)
- **Citation:** P. Howell, S. Davis, and J. Bartrip, "The UCLASS archive of stuttered speech," *J. Speech, Lang., Hear. Res.*, vol. 52, no. 2, pp. 556–569, 2009.

### Dataset 2 — Normal speech

- **Classes used:** Normal (mapped to **Non-stuttered**)
- **Purpose:** contributes fluent, non-stuttering-speaker recordings to the **Non-stuttered** class
- **Original source / link:** **[ADD INFORMATION]** — not confirmed from the supplied project files. Do not guess; add the exact source (Kaggle link, corpus name, or citation) here once identified.

### Dataset composition (after quality-control filtering)

| Category | Recordings |
|---|---|
| Normal | 8,823 |
| Fluent | 2,221 |
| Dysfluent | 2,490 |
| **Total** | **13,534** |

Of an original 13,535 raw recordings, 13,534 were retained after quality-control filtering (1 file failed processing). After speaker-identifier recovery, the corpus comprises **44 unique speakers** in total.

<a id="dataset-label-transformation"></a>
### Dataset Label Transformation

The classification target is **binary**, derived from the three source categories:

```
Normal    → Non-stuttered  (0)
Fluent    → Non-stuttered  (0)
Dysfluent → Stuttered      (1)
```

This reflects a clinically motivated distinction: both Normal recordings (non-stuttering speakers) and Fluent recordings (fluent utterances from speakers who stutter) are perceptually non-disfluent at the utterance level, while Dysfluent recordings contain audible stuttering behavior. This logic is implemented directly as a `create_binary_label` function mapping `{"normal", "fluent"} → 0` and `{"dysfluent"} → 1`.

---

##  Dataset Organization

Folder structure as referenced in the project's preprocessing/figure-generation code:

```text
dataset/
├── normal/
│   └── speakers/          # Normal-speech recordings
├── sttutering/             # (as named in the source code)
│   └── clips/              # Fluent / Dysfluent stuttering-speech clips
├── audio_dataset.csv       # Raw file manifest (label, file_path)
├── processed_dataset.csv   # After audio preprocessing
├── final_dataset.csv       # After binary label transformation
├── final_dataset_fixed.csv # After speaker-ID recovery
├── train.csv / validation.csv / test.csv   # Speaker-disjoint splits
└── features/
    ├── X_train.npy / y_train.npy
    ├── X_validation.npy / y_validation.npy
    └── X_test.npy / y_test.npy
```

Example stuttering-clip filename: `M_0030_16y4m_1_dysfluent_000.wav` (speaker code `M_0030`, parsed via a `[MF]_####` regex to recover speaker identity for speaker-independent partitioning).

---

##  Audio Preprocessing

Every recording passes through a deterministic preprocessing pipeline before feature extraction:

| Processing Stage | Value |
|---|---|
| Original dataset | 13,535 recordings |
| Files successfully processed | 13,534 recordings |
| Failed files | 1 |
| Target sample rate | 16,000 Hz |
| Audio channels | Mono |
| Silence-trim threshold (`top_db`) | 30 dB |
| Minimum duration retained | 0.20 s |
| Amplitude normalization | Peak normalization (max‑abs = 1), applied |
| NaN / infinity handling | Non-finite samples replaced with 0 |

Steps: **load & resample to 16 kHz mono → replace non-finite samples → trim leading/trailing silence (30 dB threshold) → discard clips shorter than 0.20 s → peak-normalize amplitude → persist cleaned 16 kHz mono audio.**

<img width="7200" height="5392" alt="Figure_1_Waveform_Comparison" src="https://github.com/user-attachments/assets/f51f70de-cc60-4a87-8dce-ac5ecf7a2330" />
*Temporal (waveform) characteristics of Normal, Fluent, and Dysfluent speech signals — differences in amplitude envelope and pause structure across the three original recording categories.*

<img width="6602" height="5899" alt="Figure_6_Spectrogram_Comparison" src="https://github.com/user-attachments/assets/a2a399a6-39a5-46af-8518-9c6cc4f72aa9" />
*Time–frequency (STFT spectrogram) comparison of the three original speech categories. These spectrogram/waveform visualizations are used only for illustrative acoustic analysis — they are **not** the representation the CNN receives as input (see below).*

---

##  Feature Representation

Feature extraction is a separate stage from audio cleaning. Each cleaned waveform is truncated/zero-padded to a fixed length of **5.0 s (80,000 samples at 16 kHz)**, so every recording produces an identically shaped feature tensor for batched CNN training.

| Parameter | Value |
|---|---|
| MFCC count | 40 |
| FFT size (`n_fft`) | 512 (≈32 ms window @ 16 kHz) |
| Hop length (`hop_length`) | 256 (≈16 ms hop, 50% overlap) |
| Fixed clip duration | 5.0 s (80,000 samples) |
| Derivative order | 1st (Δ) and 2nd (ΔΔ) |
| Stacked channels | 40 × 3 = 120 |
| Analysis frames | `1 + ⌊80,000 / 256⌋ = 313` |
| **Final feature tensor** | **120 × 313 (reshaped to 120 × 313 × 1)** |
| Standardization | Per-recording, per-coefficient (zero mean, unit variance) |

<img width="1804" height="872" alt="ChatGPT Image Aug 13, 2026, 12_28_20 AM" src="https://github.com/user-attachments/assets/d1b7594f-db35-4648-9239-90cc3a6966f8" />
*Representative MFCC (A), first-order delta (B), and second-order delta-delta (C) coefficients for one recording across all 313 analysis frames — together forming the 120-row feature tensor.*

<img width="1762" height="893" alt="ChatGPT Image Aug 13, 2026, 12_56_14 AM" src="https://github.com/user-attachments/assets/34cd038e-d669-4bea-9035-fe98c9ac5006" />
*Construction of the stacked 120 × 313 × 1 feature tensor from the MFCC, Δ, and ΔΔ channels, and its role as the direct input to the CNN V2 architecture.*

<img width="5147" height="4140" alt="Figure_Feature_Family_Correlation_Heatmap" src="https://github.com/user-attachments/assets/22e82cd5-fdda-498a-918b-5889c1ebcafe" />
*Pairwise Pearson correlation between the (flattened) MFCC, Δ MFCC, and ΔΔ MFCC feature families across the corpus. The static MFCC channel is essentially uncorrelated with Δ (r ≈ −0.01) and moderately anti-correlated with ΔΔ (r ≈ −0.49); Δ and ΔΔ are only weakly correlated (r ≈ −0.03) — the three channels are not simple redundant restatements of one another.*

**Note on figures:** the waveform and spectrogram images in this README are provided for **illustrative acoustic visualization only**. The CNN's actual input, verified directly in the feature-extraction code, is the 120 × 313 MFCC + Δ + ΔΔ tensor described above.

---

##  Speaker Identification & Speaker-Independent Partitioning

Speaker identity is required to guarantee that no individual's voice characteristics appear in more than one of the train/validation/test partitions.

- **Normal-speech recordings:** original speaker identifier retained directly from the dataset.
- **Stuttering-dataset recordings:** speaker identifier recovered from a structured filename prefix (e.g., `M_0030_16y4m_1_dysfluent_000.wav` → speaker code `M_0030`) via a regex match against the leading `[MF]_####` pattern.
- **Partitioning:** performed at the **speaker level**, independently within the "normal-speaker" group and the "stuttering-speaker" group, using a **70% / 15% / 15%** split (`train_test_split`, `random_state = 42`), then recombined into final train/validation/test speaker sets. All of a given speaker's recordings are assigned as a block to that speaker's partition.
- **Leakage check:** verified programmatically by computing the pairwise intersection of speaker-ID sets across all three partitions (`Train ∩ Val`, `Train ∩ Test`, `Val ∩ Test`); the split is accepted only when all three intersections are empty and both classes are present in every partition.

| Set | Recordings | Speakers | Non-stuttered | Stuttered |
|---|---|---|---|---|
| Training | 9,385 | 30 | 7,942 | 1,443 |
| Validation | 1,997 | 7 | 1,388 | 609 |
| Testing | 2,152 | 7 | 1,714 | 438 |
| **Total** | **13,534** | **44** | **11,044** | **2,490** |

Because partitioning is speaker-level rather than recording-level, class balance differs slightly by split (≈84.6% / 15.4% Non-stuttered/Stuttered in training vs. ≈79.6% / 20.4% in testing) — an expected consequence of speaker-level stratification, not a deliberately stratified split.

---

##  Model Architecture — CNN V2

`CNN V2` (saved as `cnn_v2_best.keras`) is the model whose results are reported in this project. It is a 3-block 2D CNN operating directly on the 120 × 313 × 1 MFCC+Δ+ΔΔ tensor:

| Block | Layer | Filters/Units | Kernel | Regularization | Notes |
|---|---|---|---|---|---|
| Input | — | — | — | — | Shape 120 × 313 × 1 |
| Block 1 | Conv2D → BatchNorm → MaxPool2D → SpatialDropout2D | 16 | 3×3, same, ReLU | L2 = 1e‑4 | Pool 2×2, dropout 0.20 |
| Block 2 | Conv2D → BatchNorm → MaxPool2D → SpatialDropout2D | 32 | 3×3, same, ReLU | L2 = 1e‑4 | Pool 2×2, dropout 0.25 |
| Block 3 | Conv2D → BatchNorm → MaxPool2D → SpatialDropout2D | 64 | 3×3, same, ReLU | L2 = 1e‑4 | Pool 2×2, dropout 0.30 |
| Global | GlobalAveragePooling2D | — | — | — | — |
| Dense | Dense → Dropout | 64, ReLU | — | L2 = 1e‑4 | Dropout 0.50 |
| Output | Dense | 1, Sigmoid | — | — | Predicted P(Stuttered) |

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam, learning rate = 0.0003 |
| Loss | Binary cross-entropy |
| Metrics tracked | Accuracy, Precision, Recall, AUC |
| Epochs (max) | 25 |
| Batch size | 64 |
| Early stopping | Monitor `val_auc` (max), patience = 5, restore best weights |
| LR scheduling | `ReduceLROnPlateau`, monitor `val_auc` (max), factor = 0.5, patience = 2, min LR = 1e‑6 |
| Checkpointing | `ModelCheckpoint`, monitor `val_auc` (max), best model → `cnn_v2_best.keras` |
| Random seed | 42 |
| Test-time inference batch size | 64 |
| Classification threshold | 0.5 |

> **Note:** the repository also contains an earlier model (`best_cnn_model.keras` / `final_cnn_model.keras`, referred to here as **CNN V1**) trained with `class_weight="balanced"` (`sklearn.utils.class_weight.compute_class_weight`). The results reported throughout this README and the accompanying paper are for **CNN V2**, which does not use explicit class weighting in its training call — the recall/precision asymmetry seen in its results is an emergent property of the class-imbalanced training data rather than an explicit re-weighting scheme.


---

##  Results

Evaluated on the held-out, speaker-disjoint test set (2,152 recordings, 7 speakers, none of whom appear in training or validation).

### Overall performance

| Metric | Test Performance |
|---|---|
| Accuracy | 91.54% |
| Precision (Stuttered) | 71.77% |
| Recall (Stuttered) | 96.35% |
| F1-score (Stuttered) | 82.26% |
| ROC-AUC | 96.89% |

<img width="6540" height="4140" alt="Figure_CNN_V2_Performance_Metrics" src="https://github.com/user-attachments/assets/3dd52fff-b131-4ff9-a98e-c29cad4ea445" />

*Overall test-set performance of CNN V2: accuracy, precision, recall, F1-score, and ROC-AUC.*

### Class-wise performance

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Non-stuttered | 98.98% | 90.32% | 94.45% | 1,714 |
| Stuttered | 71.77% | 96.35% | 82.26% | 438 |
| Macro average | 85.37% | 93.33% | 88.35% | 2,152 |
| Weighted average | 93.44% | 91.54% | 91.97% | 2,152 |

### Confusion matrix

<img width="4044" height="4130" alt="Figure_5_Confusion_Matrix" src="https://github.com/user-attachments/assets/6d4b482d-1412-43e0-9461-cd91d5508be8" />

*CNN V2 test-set confusion matrix. TN = 1,548, FP = 166, FN = 16, TP = 422 (positive class = Stuttered). All summary metrics were numerically re-derived from this matrix and found to be internally consistent (e.g., Accuracy = (1,548+422)/2,152 = 91.54%; Recall = 422/(422+16) = 96.35%).*

### ROC and Precision–Recall curves

<img width="6520" height="2620" alt="Figure_6_ROC_PR_curves" src="https://github.com/user-attachments/assets/7860a115-526d-44ff-b4a0-b7c0b6418256" />

*ROC curve (AUC = 0.9689) and Precision–Recall curve (AP = 0.8508) for CNN V2 on the test set. The ROC-AUC substantially exceeds the default-threshold accuracy, indicating the model's continuous probability scores separate the two classes well and that the precision/recall trade-off could be adjusted post hoc via threshold selection.*

### Training behavior

<img width="3000" height="1800" alt="training_accuracy" src="https://github.com/user-attachments/assets/ea740ce0-42ec-4b78-916f-19ab7bed073e" />

*Training vs. validation accuracy across epochs.*

<img width="3000" height="1800" alt="training_loss" src="https://github.com/user-attachments/assets/19278430-5b65-4479-9e4c-8132b895cd74" />

*Training vs. validation loss across epochs.*

**Reported as-is, not silently reconciled:** the plotted training curves show training accuracy rising smoothly toward ~96% while validation accuracy plateaus in roughly the 50%–62% range, with a non-monotonic validation loss (a pattern generally associated with overfitting or optimization instability). This is not straightforwardly consistent with the 91.54% accuracy reported on the held-out test set above. Because it could not be confirmed from the supplied materials whether the plotted training run corresponds exactly to the checkpoint (`cnn_v2_best.keras`) evaluated for the test-set metrics, this discrepancy is stated explicitly here (as in the accompanying paper, Sections 5.5 and 7) rather than resolved, and the plotted training dynamics should be interpreted with corresponding caution.

---

##  Discussion — Precision/Recall Trade-off

The model achieves strong recall (96.35%) on the minority Stuttered class — of 438 genuinely stuttered test recordings, only 16 were missed. For a screening/fluency-monitoring application, this is a desirable trade-off: a missed stuttering event (false negative) is more costly than a false positive (166 cases here), which can be reviewed and dismissed by a human evaluator. Precision (71.77%) reflects this asymmetry, plausibly influenced by the ~5.5:1 Non-stuttered:Stuttered class imbalance in the training data (7,942 vs. 1,443 recordings).

---

##  Limitations

- **Dataset scale and speaker diversity** — only 44 unique speakers total, with just 7 held out for testing and 7 for validation; performance estimates on such a small speaker-level test set carry non-trivial variance.
- **Class imbalance** — the Stuttered class is ~18% of the overall corpus, plausibly contributing to the precision/recall asymmetry.
- **Binary-label simplification** — collapsing Fluent and Normal speech into a single Non-stuttered class discards finer-grained, multi-class disfluency-type information (e.g., prolongation vs. repetition vs. block).
- **Recording provenance** — the Normal-speech and stuttering-speech recordings originate from different source collections; residual channel/recording-condition differences cannot be ruled out as a confound.
- **No cross-corpus / clinical validation** — all results are reported on a held-out partition of the same underlying corpus; no evaluation on an independent public benchmark (e.g., SEP-28k, UCLASS, FluencyBank) was performed.
- **Unreconciled training-dynamics evidence** — see [Training behavior](#training-behavior) above.
- **No ablation study** was performed (e.g., removing Δ/ΔΔ features, varying MFCC count, comparing against log-Mel-spectrogram input).
- **CNN V2 layer configuration in this README/paper reflects the code as supplied**; if the notebook is modified, re-verify architecture details against the current `model.ipynb` before reuse.

---

##  Repository Structure

**[ADD INFORMATION]** — populate with your actual repository layout once finalized, e.g.:

```text
.
├── assets/                     # Figures used in this README
├── dataset/                    # Raw / processed data, manifests, features (see Dataset Organization)
├── models/                     # Saved Keras models (cnn_v2_best.keras, cnn_v2_final.keras, ...)
├── results/                    # Generated evaluation figures / metrics
├── model.ipynb                 # Main notebook: preprocessing → features → CNN V2 training/evaluation
├── requirements.txt            # [ADD INFORMATION]
└── README.md
```

##  Setup

```bash
# [ADD INFORMATION] — confirm exact package/version pins used for this project
pip install pandas numpy librosa soundfile scikit-learn tensorflow tqdm matplotlib seaborn
```

Key libraries used in this project: `pandas`, `NumPy`, `librosa`, `soundfile`, `scikit-learn`, `TensorFlow/Keras`, `tqdm`, `matplotlib`, `seaborn`. Exact library/Python versions and hardware (CPU/GPU) used for training are **[ADD INFORMATION]** — not confirmed from the supplied materials.

##  Usage

**[ADD INFORMATION]** — add exact run instructions once the repository's script/notebook entry points are finalized (e.g., how to run preprocessing, feature extraction, training, and evaluation from `model.ipynb`).

---

##  Citation

If you use this work, please cite:

```
Aamir, L., & Maqbool, F. (2026). Speaker-Independent Stuttering Detection Using CNNs and
Cepstral Acoustic Representations. Supervised by Awais. CYBEX — School of IT Professionals
(PVT) Limited, in affiliation with The University of Faisalabad.
```

Please also cite the underlying UCLASS archive if you use the stuttering-speech data:

```
P. Howell, S. Davis, and J. Bartrip, "The UCLASS archive of stuttered speech,"
J. Speech, Lang., Hear. Res., vol. 52, no. 2, pp. 556–569, 2009.
```

---

##  Authors

- **Laiba Aamir**
- **Fatima Maqbool**
- Supervised by **Sir Awais**
- CYBEX — School of IT Professionals (PVT) Limited, in affiliation with The University of Faisalabad (2026)

##  License

**[ADD INFORMATION]** — no license file was included in the supplied materials.
