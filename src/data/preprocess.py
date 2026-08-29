"""
Deterministic audio preprocessing pipeline.

Reproduces the original notebook's preprocessing stage exactly:
  1. Load audio and resample to a fixed target sample rate (mono).
  2. Replace non-finite (NaN/inf) samples with zero.
  3. Trim leading/trailing silence (librosa.effects.trim, top_db=30).
  4. Discard recordings shorter than MIN_DURATION_SECONDS after trimming.
  5. Peak-amplitude normalize (max(|x|) == 1).
  6. Persist the cleaned waveform as a new 16 kHz mono WAV file.

No preprocessing logic was changed from the original implementation --
only the hard-coded local paths were replaced with configurable ones
(see config/config.py) and the code was wrapped into reusable
functions with logging instead of ad-hoc prints.
"""

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config.config import MIN_DURATION_SECONDS, SAMPLE_RATE, SILENCE_TOP_DB

logger = logging.getLogger(__name__)


def preprocess_audio(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int = SAMPLE_RATE,
    min_duration: float = MIN_DURATION_SECONDS,
    top_db: float = SILENCE_TOP_DB,
) -> Tuple[bool, str]:
    """Clean a single audio file and write the result to `output_path`.

    Returns (success, reason). `reason` is "success" on success, or a
    short failure code / exception string otherwise (mirrors the
    original notebook's return convention so downstream reporting code
    is unchanged).
    """
    try:
        audio, _ = librosa.load(input_path, sr=target_sample_rate, mono=True)

        if audio is None or len(audio) == 0:
            return False, "empty_audio"

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        audio, _ = librosa.effects.trim(audio, top_db=top_db)

        duration = len(audio) / target_sample_rate
        if duration < min_duration:
            return False, "too_short"

        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0:
            audio = audio / max_amplitude

        sf.write(output_path, audio, target_sample_rate)

        return True, "success"

    except Exception as exc:  # noqa: BLE001 - mirrors original broad catch
        return False, str(exc)


def preprocess_dataset(
    metadata_csv: Path,
    processed_audio_dir: Path,
    output_csv: Path,
    failed_csv: Path,
) -> pd.DataFrame:
    """Run preprocess_audio() over every row of the raw metadata CSV.

    For each successfully processed file: writes a cleaned WAV into
    `processed_audio_dir`, and records the new `file_path`,
    `original_path`, `sample_rate`, and `channels` in the returned
    dataframe (which is also written to `output_csv`). Failures are
    logged to `failed_csv` with a reason code.
    """
    processed_audio_dir = Path(processed_audio_dir)
    processed_audio_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    logger.info("Preprocessing %d audio files from %s", len(df), metadata_csv)
    logger.info("Original label distribution:\n%s", df["label"].value_counts())

    processed_rows = []
    failed_files = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio"):
        input_path = Path(row["file_path"])

        if not input_path.exists():
            failed_files.append({"file": str(input_path), "reason": "file_not_found"})
            continue

        output_filename = f"{index:06d}_{input_path.stem}.wav"
        output_path = processed_audio_dir / output_filename

        success, reason = preprocess_audio(input_path, output_path)

        if success:
            new_row = row.copy()
            new_row["original_path"] = str(input_path)
            new_row["file_path"] = str(output_path)
            new_row["sample_rate"] = SAMPLE_RATE
            new_row["channels"] = 1
            new_row["processed"] = True
            processed_rows.append(new_row)
        else:
            failed_files.append({"file": str(input_path), "reason": reason})

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(output_csv, index=False)

    failed_df = pd.DataFrame(failed_files)
    failed_df.to_csv(failed_csv, index=False)

    logger.info(
        "Preprocessing complete: %d succeeded, %d failed",
        len(processed_df),
        len(failed_df),
    )
    if len(failed_df) > 0:
        logger.info("Failure reasons:\n%s", failed_df["reason"].value_counts())

    return processed_df
