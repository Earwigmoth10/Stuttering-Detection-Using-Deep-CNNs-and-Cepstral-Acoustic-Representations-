#!/usr/bin/env python3
"""
Stage 3: Load a trained checkpoint and evaluate it on the held-out,
speaker-disjoint test set.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model-path models/cnn_v2_best.keras --prefix cnn_v2
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import config  # noqa: E402
from src.evaluation.evaluate import evaluate_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(config.MODEL_DIR / "cnn_v2_best.keras"),
        help="Path to a .keras model checkpoint (default: models/cnn_v2_best.keras).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="cnn_v2",
        help="Prefix used for output figure/metric filenames.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(
            "Model checkpoint not found: %s\nRun scripts/train_model.py first.",
            model_path,
        )
        sys.exit(1)

    x_test_path = config.FEATURE_DIR / "X_test.npy"
    y_test_path = config.FEATURE_DIR / "y_test.npy"
    if not x_test_path.exists() or not y_test_path.exists():
        logger.error(
            "Test features not found under %s.\nRun scripts/prepare_dataset.py "
            "and scripts/train_model.py (which extracts features) first.",
            config.FEATURE_DIR,
        )
        sys.exit(1)

    logger.info("Loading model: %s", model_path)
    model = tf.keras.models.load_model(model_path)

    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    logger.info("Test set: X=%s y=%s", X_test.shape, y_test.shape)

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        results_dir=config.RESULTS_DIR,
        prefix=args.prefix,
    )

    logger.info("Final test metrics: %s", metrics)


if __name__ == "__main__":
    main()
