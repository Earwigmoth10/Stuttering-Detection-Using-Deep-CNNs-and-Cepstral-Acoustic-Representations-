"""
Dataset loading, label construction, and speaker-identification utilities.

This module reproduces (as reusable functions) the logic originally
written across several exploratory notebook cells:
  - loading the raw audio_dataset.csv metadata,
  - deriving the original three-way recording category
    (normal / fluent / dysfluent) from the `label` column and filename,
  - collapsing the three-way category into the binary target used for
    classification (Normal, Fluent -> 0 = non_stuttered;
    Dysfluent -> 1 = stuttered),
  - recovering a per-recording speaker identifier (direct for the
    normal-speech set, parsed from a `[MF]_####` filename prefix for
    the stuttering-dataset set),
  - reporting class/speaker distribution.

No feature values, thresholds, or label-mapping rules were changed
from the original notebook implementation.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SPEAKER_ID_PATTERN = re.compile(r"^([MF]_\d+)_")


# ============================================================
# LOADING
# ============================================================

def load_raw_metadata(csv_path: Path) -> pd.DataFrame:
    """Load the raw dataset metadata CSV (label, file_path, speaker_id, ...).

    Raises FileNotFoundError with a helpful message if the CSV (and by
    extension the dataset) has not been configured yet -- see
    data/README.md for how to obtain and place the dataset.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata CSV not found at: {csv_path}\n"
            "Set the STUTTERING_DATASET_DIR environment variable (or "
            "edit config/config.py) to point at your local copy of the "
            "dataset. See data/README.md for details."
        )

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)
    return df


def summarize_raw_dataset(df: pd.DataFrame) -> dict:
    """Report top-level composition of the raw (label-only) dataset."""
    normal_data = df[df["label"] == "normal"]
    stuttering_data = df[df["label"] == "stuttering"]

    return {
        "total_files": len(df),
        "normal_files": len(normal_data),
        "stuttering_files": len(stuttering_data),
        "label_counts": df["label"].value_counts().to_dict(),
    }


# ============================================================
# ORIGINAL CATEGORY / FILENAME PARSING
# ============================================================

def get_filename_category(filename: str) -> str:
    """Classify a stuttering-set filename as fluent/dysfluent/other.

    Mirrors the original `get_category` helper: a case-insensitive
    substring match on "dysfluent" / "fluent" within the filename.
    """
    filename = str(filename).lower()

    if "dysfluent" in filename:
        return "dysfluent"

    if "fluent" in filename:
        return "fluent"

    return "other"


def get_original_category(row: pd.Series) -> str:
    """Determine the original three-way recording category for a row.

    - Rows from the normal-speech source (`label == "normal"`) are
      category "normal".
    - Rows from the stuttering-speech source are categorized as
      "dysfluent" or "fluent" based on their filename, or "unknown" if
      neither substring is present.
    """
    if row["label"] == "normal":
        return "normal"

    filename = str(row["file_name"]).lower()

    if "dysfluent" in filename:
        return "dysfluent"

    if "fluent" in filename:
        return "fluent"

    return "unknown"


# ============================================================
# BINARY LABEL CONSTRUCTION
# ============================================================

def create_binary_label(category: str) -> int:
    """Map the original three-way category to the binary classification
    target used throughout this project.

        Normal, Fluent -> 0 (non_stuttered)
        Dysfluent      -> 1 (stuttered)

    Returns -1 for any other/unrecognized category (these rows should
    be filtered out before training; see build_final_dataset()).
    """
    if category in ("normal", "fluent"):
        return 0

    if category == "dysfluent":
        return 1

    return -1


BINARY_LABEL_NAMES = {0: "non_stuttered", 1: "stuttered"}

SOURCE_BY_CATEGORY = {
    "normal": "normal_speakers",
    "fluent": "stuttering_dataset",
    "dysfluent": "stuttering_dataset",
}


# ============================================================
# SPEAKER IDENTIFICATION
# ============================================================

def fix_speaker_id(row: pd.Series) -> str:
    """Recover a per-recording speaker identifier.

    - Normal-speech rows keep their original `speaker_id` as supplied
      with the dataset.
    - Stuttering-dataset rows have their speaker code parsed from a
      leading "[MF]_####_" filename prefix (e.g.
      "M_0030_16y4m_1_dysfluent_000.wav" -> "M_0030"). If the pattern
      does not match, the original speaker_id value is kept as a
      fallback.
    """
    category = row["original_category"]

    if category == "normal":
        return row["speaker_id"]

    filename = str(row["file_name"])
    match = SPEAKER_ID_PATTERN.match(filename)

    if match:
        return match.group(1)

    return row["speaker_id"]


def build_speaker_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-speaker statistics used for speaker-independent
    partitioning: total recordings, dysfluent-recording count, and a
    derived speaker_type ("stuttering" if the speaker has at least one
    dysfluent recording, else "normal").
    """
    speaker_info = (
        df.groupby("speaker_id")
        .agg(
            total_recordings=("binary_label", "count"),
            dysfluent_recordings=("binary_label", "sum"),
        )
        .reset_index()
    )

    speaker_info["speaker_type"] = speaker_info["dysfluent_recordings"].apply(
        lambda x: "stuttering" if x > 0 else "normal"
    )

    return speaker_info


# ============================================================
# FULL PIPELINE: RAW -> FINAL LABELED DATASET
# ============================================================

def build_final_dataset(
    processed_csv: Path,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Reproduce the "prepare final dataset" + "fix speaker IDs" stages.

    Takes the CSV produced by the audio-preprocessing stage (one row
    per successfully cleaned recording, with `label` and `file_name`
    columns) and returns a dataframe with:
      original_category, binary_label, binary_label_name, source,
      and a corrected speaker_id.

    Rows whose original_category cannot be determined ("unknown") are
    dropped, with a warning, exactly as in the original notebook.
    """
    df = load_raw_metadata(processed_csv)

    df["original_category"] = df.apply(get_original_category, axis=1)

    unknown = df[df["original_category"] == "unknown"]
    if len(unknown) > 0:
        logger.warning(
            "%d recordings had an unresolved ('unknown') category and "
            "were dropped: %s",
            len(unknown),
            unknown["file_name"].head(20).tolist(),
        )
        df = df[df["original_category"] != "unknown"].copy()

    df["binary_label"] = df["original_category"].apply(create_binary_label)
    df["binary_label_name"] = df["binary_label"].map(BINARY_LABEL_NAMES)
    df["source"] = df["original_category"].map(SOURCE_BY_CATEGORY)

    final_df = df[
        [
            "file_path",
            "file_name",
            "speaker_id",
            "original_category",
            "binary_label",
            "binary_label_name",
            "source",
        ]
    ].copy()

    # Fix speaker IDs (requires original_category, computed above)
    df_for_speaker_fix = df.copy()
    final_df["speaker_id"] = df_for_speaker_fix.apply(fix_speaker_id, axis=1)

    final_df = final_df.sample(frac=1, random_state=random_seed).reset_index(
        drop=True
    )

    return final_df


def validate_dataset_structure(dataset_dir: Path) -> Optional[str]:
    """Sanity-check that the configured dataset directory looks usable.

    Returns None if the directory looks fine, otherwise a human
    readable explanation of what's missing (used by
    scripts/prepare_dataset.py to fail fast with a clear message).
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        return f"Dataset directory does not exist: {dataset_dir}"

    csv_file = dataset_dir / "audio_dataset.csv"
    if not csv_file.exists():
        return (
            f"Expected metadata CSV not found: {csv_file}\n"
            "See data/README.md for the expected dataset layout."
        )

    return None
