"""
Centralized configuration for the stuttering-detection project.

All machine-specific values (dataset location, output directories) are
resolved relative to the project root or from environment variables /
a `.env` file. Nothing in this file should ever contain a personal,
machine-specific absolute path — see SECURITY_AUDIT.md for the paths
that were removed from the original notebook and replaced with the
configuration system defined here.

Precedence for locating the raw dataset (highest wins):
    1. Explicit CLI argument passed to a script (e.g. --dataset-dir)
    2. Environment variable `STUTTERING_DATASET_DIR`
    3. `.env` file value for `DATASET_ROOT` (loaded via python-dotenv)
    4. Default fallback: <project_root>/data/raw
"""

import os
from pathlib import Path

try:
    # Optional: if python-dotenv is installed and a .env file exists,
    # values defined there populate os.environ. This is entirely
    # optional -- the project works without a .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DATA DIRECTORIES
# ============================================================

# Root directory holding the raw dataset (audio + metadata CSV).
# Configurable via the STUTTERING_DATASET_DIR environment variable or a
# `.env` file (see .env.example). Falls back to a project-relative
# folder so the code runs out-of-the-box on a fresh clone, provided the
# researcher places (or symlinks) their dataset there.
DATASET_ROOT = Path(
    os.environ.get("STUTTERING_DATASET_DIR")
    or os.environ.get("DATASET_ROOT")
    or (PROJECT_ROOT / "data" / "raw")
).resolve()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATASET_ROOT
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Metadata CSVs
AUDIO_METADATA_CSV = DATASET_ROOT / "audio_dataset.csv"
PROCESSED_METADATA_CSV = PROCESSED_DATA_DIR / "processed_dataset.csv"
FAILED_FILES_CSV = PROCESSED_DATA_DIR / "failed_files.csv"
FINAL_METADATA_CSV = PROCESSED_DATA_DIR / "final_dataset.csv"
FINAL_METADATA_FIXED_CSV = PROCESSED_DATA_DIR / "final_dataset_fixed.csv"

TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_CSV = PROCESSED_DATA_DIR / "validation.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

# Cleaned/resampled audio produced by the preprocessing stage
PROCESSED_AUDIO_DIR = PROCESSED_DATA_DIR / "processed_audio"

# Extracted MFCC(+delta+delta-delta) feature tensors
FEATURE_DIR = PROCESSED_DATA_DIR / "features"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

for _dir in (PROCESSED_DATA_DIR, PROCESSED_AUDIO_DIR, FEATURE_DIR,
             MODEL_DIR, FIGURES_DIR, METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO PREPROCESSING PARAMETERS
# (as implemented in the original notebook, Section "Preprocess Audio")
# ============================================================

SAMPLE_RATE = 16_000          # Target sampling rate (Hz), mono
MIN_DURATION_SECONDS = 0.20   # Recordings shorter than this after
                               # silence-trimming are discarded
SILENCE_TOP_DB = 30           # librosa.effects.trim threshold (dB)


# ============================================================
# FEATURE EXTRACTION PARAMETERS
# (as implemented in the original notebook, Section "Feature Extraction")
# ============================================================

N_MFCC = 40                   # Number of MFCC coefficients
N_FFT = 512                   # FFT window size
HOP_LENGTH = 256               # Hop length between STFT frames
MAX_DURATION_SECONDS = 5.0     # Fixed-length window used for all clips
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_SECONDS)

# Resulting feature tensor shape: (N_MFCC * 3, n_frames) -> (120, 313)


# ============================================================
# DATASET SPLITTING
# ============================================================

RANDOM_SEED = 42
SPEAKER_TEST_SIZE = 0.30       # First split: 70% train / 30% temp
SPEAKER_VAL_TEST_SIZE = 0.50   # Second split of the 30% temp: 15/15


# ============================================================
# TRAINING PARAMETERS
# ============================================================
#
# Two model configurations exist in the original notebook:
#
#   CNN V1 ("model.py: build_cnn_v1"): the first architecture that was
#   trained and evaluated in the notebook.
#
#   CNN V2 ("model.py: build_cnn_v2"): a smaller, L2-regularized
#   architecture with SpatialDropout, trained afterwards. This is the
#   version referenced throughout the accompanying research paper
#   (checkpoint file name `cnn_v2_best.keras`) and the one whose
#   metrics (Accuracy 91.54%, F1 82.26%, ROC-AUC 96.89%) are reported
#   as the project's final results.
#
# Both configurations are preserved exactly as implemented. Scripts in
# this project default to CNN V2 since it is the version whose results
# are reported, but CNN V1 remains available via --model-version v1.

class CNNV1TrainingConfig:
    BATCH_SIZE = 32
    EPOCHS = 40
    OPTIMIZER = "adam"
    LEARNING_RATE = 0.001
    LOSS = "binary_crossentropy"
    EARLY_STOPPING_MONITOR = "val_loss"
    EARLY_STOPPING_PATIENCE = 8
    REDUCE_LR_MONITOR = "val_loss"
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_MIN_LR = 1e-6
    CHECKPOINT_MONITOR = "val_loss"
    CHECKPOINT_FILENAME = "best_cnn_model.keras"
    FINAL_MODEL_FILENAME = "final_cnn_model.keras"


class CNNV2TrainingConfig:
    BATCH_SIZE = 64
    EPOCHS = 25
    OPTIMIZER = "adam"
    LEARNING_RATE = 0.0003
    LOSS = "binary_crossentropy"
    EARLY_STOPPING_MONITOR = "val_auc"
    EARLY_STOPPING_MODE = "max"
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_MONITOR = "val_auc"
    REDUCE_LR_MODE = "max"
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 2
    REDUCE_LR_MIN_LR = 1e-6
    CHECKPOINT_MONITOR = "val_auc"
    CHECKPOINT_MODE = "max"
    CHECKPOINT_FILENAME = "cnn_v2_best.keras"
    FINAL_MODEL_FILENAME = "cnn_v2_final.keras"


# Test-time inference batch size (both model versions, as in the
# original evaluation script)
INFERENCE_BATCH_SIZE = 64

# Classification decision threshold applied to the sigmoid output
CLASSIFICATION_THRESHOLD = 0.5
