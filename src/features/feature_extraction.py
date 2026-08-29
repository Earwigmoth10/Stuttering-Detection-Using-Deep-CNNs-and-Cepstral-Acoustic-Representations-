"""
Cepstral (MFCC + delta + delta-delta) feature extraction.

Feature representation (unchanged from the original notebook):
  - 40 MFCCs (n_mfcc=40) computed with n_fft=512, hop_length=256 from
    audio resampled to 16 kHz.
  - First-order (delta) and second-order (delta-delta) time
    derivatives of the MFCCs, stacked with the static coefficients:
    features = vstack([mfcc, delta, delta2]) -> 120 rows.
  - Each waveform is silence-trimmed, then truncated/zero-padded to a
    fixed 5.0 s window (80,000 samples @ 16 kHz) before MFCC
    computation, so every recording yields a (120, 313) tensor.
  - Peak-amplitude normalization is applied a second time at this
    stage (in addition to the preprocessing stage), matching the
    original implementation.
  - Each of the 120 feature rows is standardized independently
    (zero mean, unit variance, per-recording) across its own time axis.

Final per-recording feature tensor shape: (120, 313), float32.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import HOP_LENGTH, MAX_SAMPLES, N_FFT, N_MFCC, SAMPLE_RATE

logger = logging.getLogger(__name__)


def extract_features(audio_path: str) -> Optional[np.ndarray]:
    """Extract the (120, 313) MFCC+delta+delta-delta feature tensor for
    one audio file. Returns None (and logs the error) on failure,
    mirroring the original notebook's error handling so a single bad
    file does not stop a full dataset run.
    """
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        audio, _ = librosa.effects.trim(audio, top_db=30)

        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        else:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)), mode="constant")

        max_value = np.max(np.abs(audio))
        if max_value > 0:
            audio = audio / max_value

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.vstack([mfcc, delta, delta2])

        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        features = (features - mean) / (std + 1e-8)

        return features.astype(np.float32)

    except Exception as exc:  # noqa: BLE001 - mirrors original broad catch
        logger.error("Feature extraction failed for %s: %s", audio_path, exc)
        return None


def extract_features_for_split(
    csv_path: Path,
    feature_dir: Path,
    feature_filename: str,
    label_filename: str,
    failed_filename: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for every recording listed in a split CSV
    (train.csv / validation.csv / test.csv) and persist the resulting
    (X, y) arrays as .npy files under `feature_dir`.
    """
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    logger.info("Extracting features for %d files from %s", len(df), csv_path)

    features: List[np.ndarray] = []
    labels: List[int] = []
    failed_files: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=Path(csv_path).stem):
        audio_path = str(row["file_path"])
        feature = extract_features(audio_path)

        if feature is None:
            failed_files.append(audio_path)
            continue

        features.append(feature)
        labels.append(int(row["binary_label"]))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    np.save(feature_dir / feature_filename, X)
    np.save(feature_dir / label_filename, y)

    if failed_files and failed_filename:
        pd.DataFrame({"file_path": failed_files}).to_csv(
            feature_dir / failed_filename, index=False
        )

    logger.info(
        "Done: %s -> X %s, y %s (non_stuttered=%d, stuttered=%d, failed=%d)",
        Path(csv_path).name,
        X.shape,
        y.shape,
        int(np.sum(y == 0)) if len(y) else 0,
        int(np.sum(y == 1)) if len(y) else 0,
        len(failed_files),
    )

    return X, y
