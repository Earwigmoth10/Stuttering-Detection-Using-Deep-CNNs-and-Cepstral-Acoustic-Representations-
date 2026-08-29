"""
Evaluation utilities for the CNN stuttering-detection models.

Reproduces the original notebook's evaluation logic exactly:
  - predicted probability = model.predict(X_test).ravel()
  - predicted class = probability >= 0.5 (config.CLASSIFICATION_THRESHOLD)
  - accuracy, precision, recall, F1 (positive class = "stuttered" = 1)
  - ROC-AUC (from the continuous probability, not the thresholded class)
  - full sklearn classification_report and confusion matrix
  - training/validation loss, accuracy, and (for CNN V2) AUC curves
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for script/CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.config import CLASSIFICATION_THRESHOLD, INFERENCE_BATCH_SIZE

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Non-Stuttered", "Stuttered"]


def predict(model: tf.keras.Model, X_test: np.ndarray) -> np.ndarray:
    if X_test.ndim == 3:
        X_test = X_test[..., np.newaxis]
    return model.predict(X_test, batch_size=INFERENCE_BATCH_SIZE, verbose=1).ravel()


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def print_classification_report(y_true: np.ndarray, probabilities: np.ndarray) -> str:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    report = classification_report(
        y_true, predictions, target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    logger.info("\n%s", report)
    return report


def plot_confusion_matrix(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    title: str = "CNN Confusion Matrix",
) -> np.ndarray:
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return cm


def plot_training_curves(
    history: tf.keras.callbacks.History, output_dir: Path, prefix: str = "cnn"
) -> None:
    """Save training/validation loss, accuracy, and (if present) AUC
    curves, matching the plots generated in the original notebook.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["loss"], label="Training Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix.upper()} Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["accuracy"], label="Training Accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix.upper()} Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_accuracy.png", dpi=200)
    plt.close()

    if "auc" in history_df.columns and "val_auc" in history_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history_df["auc"], label="Training AUC")
        plt.plot(history_df["val_auc"], label="Validation AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{prefix.upper()} Training vs Validation AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_auc.png", dpi=200)
        plt.close()


def save_metrics(metrics: Dict[str, float], output_csv: Path) -> None:
    pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    ).to_csv(output_csv, index=False)


def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path,
    prefix: str = "cnn",
    history: Optional[tf.keras.callbacks.History] = None,
) -> Dict[str, float]:
    """Full evaluation: predictions, metrics, classification report,
    confusion matrix figure, and (if provided) training-curve figures.
    Returns the metrics dict and writes it to
    results_dir/metrics/{prefix}_metrics.csv.
    """
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    probabilities = predict(model, X_test)
    metrics = compute_metrics(y_test, probabilities)

    logger.info("Test metrics (%s): %s", prefix, metrics)
    print_classification_report(y_test, probabilities)

    plot_confusion_matrix(
        y_test,
        probabilities,
        figures_dir / f"{prefix}_confusion_matrix.png",
        title=f"{prefix.upper()} Confusion Matrix",
    )

    if history is not None:
        plot_training_curves(history, figures_dir, prefix=prefix)

    save_metrics(metrics, metrics_dir / f"{prefix}_metrics.csv")

    return metrics
