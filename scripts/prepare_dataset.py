#!/usr/bin/env python3
"""
Stage 1: Dataset preparation.

Runs, in order:
  1. Audio preprocessing (resample, trim, normalize, persist cleaned WAVs).
  2. Binary-label construction + speaker-ID recovery.
  3. Speaker-independent train/validation/test split (70/15/15, split
     separately within the normal-speaker and stuttering-speaker
     groups, then recombined), with a programmatic leakage check.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dataset-dir /path/to/dataset
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from config import config  # noqa: E402
from src.data.load_dataset import (  # noqa: E402
    build_final_dataset,
    build_speaker_table,
    validate_dataset_structure,
)
from src.data.preprocess import preprocess_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_speakers_independently(
    df: pd.DataFrame, seed: int = config.RANDOM_SEED
):
    """Speaker-independent 70/15/15 split, performed separately within
    the normal-speaker and stuttering-speaker groups and then
    recombined -- exactly as in the original notebook.
    """
    speaker_info = build_speaker_table(df)

    normal_speakers = speaker_info[speaker_info["speaker_type"] == "normal"][
        "speaker_id"
    ].tolist()
    stuttering_speakers = speaker_info[speaker_info["speaker_type"] == "stuttering"][
        "speaker_id"
    ].tolist()

    normal_train, normal_temp = train_test_split(
        normal_speakers, test_size=config.SPEAKER_TEST_SIZE, random_state=seed
    )
    normal_val, normal_test = train_test_split(
        normal_temp, test_size=config.SPEAKER_VAL_TEST_SIZE, random_state=seed
    )

    stutter_train, stutter_temp = train_test_split(
        stuttering_speakers, test_size=config.SPEAKER_TEST_SIZE, random_state=seed
    )
    stutter_val, stutter_test = train_test_split(
        stutter_temp, test_size=config.SPEAKER_VAL_TEST_SIZE, random_state=seed
    )

    train_speakers = set(normal_train + stutter_train)
    val_speakers = set(normal_val + stutter_val)
    test_speakers = set(normal_test + stutter_test)

    train_df = df[df["speaker_id"].isin(train_speakers)].sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    val_df = df[df["speaker_id"].isin(val_speakers)].sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    test_df = df[df["speaker_id"].isin(test_speakers)].sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    # Programmatic leakage check
    train_ids, val_ids, test_ids = (
        set(train_df["speaker_id"]),
        set(val_df["speaker_id"]),
        set(test_df["speaker_id"]),
    )
    overlaps = {
        "train_val": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "val_test": val_ids & test_ids,
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Speaker leakage detected across splits: {overlaps}")
    logger.info("Speaker leakage check passed: zero overlap across all splits.")

    for name, split_df in (("train", train_df), ("validation", val_df), ("test", test_df)):
        classes = set(split_df["binary_label"])
        status = "OK" if classes == {0, 1} else "MISSING A CLASS"
        logger.info(
            "%s: %d recordings, %d speakers, classes=%s [%s]",
            name,
            len(split_df),
            split_df["speaker_id"].nunique(),
            classes,
            status,
        )

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Prepare the stuttering-detection dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the raw dataset directory (overrides config/.env).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.RAW_DATA_DIR

    error = validate_dataset_structure(dataset_dir)
    if error:
        logger.error(error)
        sys.exit(1)

    # 1. Preprocess audio
    logger.info("=" * 70)
    logger.info("STEP 1: AUDIO PREPROCESSING")
    logger.info("=" * 70)
    preprocess_dataset(
        metadata_csv=dataset_dir / "audio_dataset.csv",
        processed_audio_dir=config.PROCESSED_AUDIO_DIR,
        output_csv=config.PROCESSED_METADATA_CSV,
        failed_csv=config.FAILED_FILES_CSV,
    )

    # 2. Build final labeled dataset (binary label + speaker IDs)
    logger.info("=" * 70)
    logger.info("STEP 2: LABEL CONSTRUCTION & SPEAKER ID RECOVERY")
    logger.info("=" * 70)
    final_df = build_final_dataset(config.PROCESSED_METADATA_CSV)
    final_df.to_csv(config.FINAL_METADATA_FIXED_CSV, index=False)
    logger.info("Final dataset: %d recordings, %d speakers", len(final_df),
                final_df["speaker_id"].nunique())
    logger.info("Binary label distribution:\n%s",
                final_df["binary_label_name"].value_counts())

    # 3. Speaker-independent split
    logger.info("=" * 70)
    logger.info("STEP 3: SPEAKER-INDEPENDENT TRAIN/VAL/TEST SPLIT")
    logger.info("=" * 70)
    train_df, val_df, test_df = split_speakers_independently(final_df)

    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VALIDATION_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    logger.info("Saved: %s, %s, %s", config.TRAIN_CSV, config.VALIDATION_CSV, config.TEST_CSV)
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
