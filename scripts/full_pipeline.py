#!/usr/bin/env python3
"""
End-to-end reference pipeline: dataset loading -> cleaning ->
preprocessing -> feature extraction -> speaker-independent splitting ->
model creation -> training -> evaluation -> results.

This script is a single-file orchestration of the full workflow; the
actual logic lives in src/ and is imported here rather than duplicated.
It contains no personal paths, credentials, or machine-specific
values -- everything is sourced from config/config.py, which itself
resolves the dataset location from an environment variable / .env file
(see .env.example) with a project-relative fallback.

Usage:
    python scripts/full_pipeline.py
    python scripts/full_pipeline.py --model-version v1 --dataset-dir /path/to/dataset
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config  # noqa: E402
from src.data.load_dataset import (  # noqa: E402
    build_final_dataset,
    validate_dataset_structure,
)
from src.data.preprocess import preprocess_dataset  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402
from src.features.feature_extraction import extract_features_for_split  # noqa: E402
from src.training.train import train_model  # noqa: E402

# Reuse the speaker-independent splitting logic defined in prepare_dataset.py
from prepare_dataset import split_speakers_independently  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the full stuttering-detection pipeline.")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.RAW_DATA_DIR

    error = validate_dataset_structure(dataset_dir)
    if error:
        logger.error(error)
        sys.exit(1)

    # ---------------------------------------------------------------
    # 1-2. Dataset loading + cleaning/preprocessing
    # ---------------------------------------------------------------
    logger.info("STEP 1-2: Loading & preprocessing audio")
    preprocess_dataset(
        metadata_csv=dataset_dir / "audio_dataset.csv",
        processed_audio_dir=config.PROCESSED_AUDIO_DIR,
        output_csv=config.PROCESSED_METADATA_CSV,
        failed_csv=config.FAILED_FILES_CSV,
    )

    # ---------------------------------------------------------------
    # 3. Label construction + speaker ID recovery
    # ---------------------------------------------------------------
    logger.info("STEP 3: Building final labeled dataset")
    final_df = build_final_dataset(config.PROCESSED_METADATA_CSV)
    final_df.to_csv(config.FINAL_METADATA_FIXED_CSV, index=False)

    # ---------------------------------------------------------------
    # 4. Speaker-independent dataset splitting
    # ---------------------------------------------------------------
    logger.info("STEP 4: Speaker-independent train/validation/test split")
    train_df, val_df, test_df = split_speakers_independently(final_df)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    val_df.to_csv(config.VALIDATION_CSV, index=False)
    test_df.to_csv(config.TEST_CSV, index=False)

    # ---------------------------------------------------------------
    # 5. Feature extraction
    # ---------------------------------------------------------------
    logger.info("STEP 5: Feature extraction (MFCC + delta + delta-delta)")
    X_train, y_train = extract_features_for_split(
        config.TRAIN_CSV, config.FEATURE_DIR, "X_train.npy", "y_train.npy", "train_failed.csv"
    )
    X_val, y_val = extract_features_for_split(
        config.VALIDATION_CSV,
        config.FEATURE_DIR,
        "X_validation.npy",
        "y_validation.npy",
        "validation_failed.csv",
    )
    X_test, y_test = extract_features_for_split(
        config.TEST_CSV, config.FEATURE_DIR, "X_test.npy", "y_test.npy", "test_failed.csv"
    )

    # ---------------------------------------------------------------
    # 6-7. Model creation + training
    # ---------------------------------------------------------------
    logger.info("STEP 6-7: Model creation & training (%s)", args.model_version)
    model, history, checkpoint_path, final_model_path = train_model(
        version=args.model_version,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir=config.MODEL_DIR,
    )

    # ---------------------------------------------------------------
    # 8-9. Evaluation + results
    # ---------------------------------------------------------------
    logger.info("STEP 8-9: Evaluation on the held-out test set")
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        results_dir=config.RESULTS_DIR,
        prefix=f"cnn_{args.model_version}",
        history=history,
    )

    logger.info("Pipeline complete. Final test metrics: %s", metrics)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Final model: %s", final_model_path)


if __name__ == "__main__":
    main()
