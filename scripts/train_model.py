#!/usr/bin/env python3
"""
Stage 2: Feature extraction (if needed) + model training.

Usage:
    python scripts/train_model.py                 # trains CNN V2 (default,
                                                    # matches reported results)
    python scripts/train_model.py --model-version v1
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config import config  # noqa: E402
from src.features.feature_extraction import extract_features_for_split  # noqa: E402
from src.training.train import train_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_or_extract_features():
    splits = {
        "train": (config.TRAIN_CSV, "X_train.npy", "y_train.npy", "train_failed.csv"),
        "validation": (
            config.VALIDATION_CSV,
            "X_validation.npy",
            "y_validation.npy",
            "validation_failed.csv",
        ),
        "test": (config.TEST_CSV, "X_test.npy", "y_test.npy", "test_failed.csv"),
    }

    data = {}
    for name, (csv_path, x_name, y_name, failed_name) in splits.items():
        x_path = config.FEATURE_DIR / x_name
        y_path = config.FEATURE_DIR / y_name

        if x_path.exists() and y_path.exists():
            logger.info("Loading cached features for %s", name)
            X, y = np.load(x_path), np.load(y_path)
        else:
            if not csv_path.exists():
                raise FileNotFoundError(
                    f"{csv_path} not found. Run scripts/prepare_dataset.py first."
                )
            X, y = extract_features_for_split(
                csv_path, config.FEATURE_DIR, x_name, y_name, failed_name
            )
        data[name] = (X, y)

    return data


def main():
    parser = argparse.ArgumentParser(description="Train the stuttering-detection CNN.")
    parser.add_argument(
        "--model-version",
        choices=["v1", "v2"],
        default="v2",
        help="Which architecture to train (default: v2, the reported model).",
    )
    args = parser.parse_args()

    data = load_or_extract_features()
    X_train, y_train = data["train"]
    X_val, y_val = data["validation"]

    logger.info("Train: X=%s y=%s | Validation: X=%s y=%s",
                X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    model, history, checkpoint_path, final_model_path = train_model(
        version=args.model_version,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_dir=config.MODEL_DIR,
    )

    logger.info("Training complete.")
    logger.info("Best checkpoint: %s", checkpoint_path)
    logger.info("Final model: %s", final_model_path)


if __name__ == "__main__":
    main()
