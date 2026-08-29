"""
Model training for the CNN stuttering-detection models.

Preserves, exactly, the optimizer, loss, learning rate, batch size,
epoch count, callbacks, and class-weighting strategy found in the
original notebook for each model version:

  CNN V1: Adam(lr=0.001), binary_crossentropy, batch_size=32,
  epochs=40, class_weight="balanced" (sklearn compute_class_weight),
  EarlyStopping(monitor="val_loss", patience=8,
  restore_best_weights=True), ReduceLROnPlateau(monitor="val_loss",
  factor=0.5, patience=3, min_lr=1e-6), ModelCheckpoint(monitor=
  "val_loss", save_best_only=True).

  CNN V2: Adam(lr=0.0003), binary_crossentropy, batch_size=64,
  epochs=25, no explicit class_weight argument (unlike V1),
  EarlyStopping(monitor="val_auc", mode="max", patience=5,
  restore_best_weights=True), ReduceLROnPlateau(monitor="val_auc",
  mode="max", factor=0.5, patience=2, min_lr=1e-6),
  ModelCheckpoint(monitor="val_auc", mode="max", save_best_only=True).

Random seeds (numpy and TensorFlow) are fixed to config.RANDOM_SEED
(42) before model construction, matching the original notebook.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config.config import (
    CNNV1TrainingConfig,
    CNNV2TrainingConfig,
    RANDOM_SEED,
)
from src.models.model import build_model

logger = logging.getLogger(__name__)


def set_seeds(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_balanced_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def _build_callbacks_v1(checkpoint_path: Path) -> list:
    cfg = CNNV1TrainingConfig
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=cfg.EARLY_STOPPING_MONITOR,
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.REDUCE_LR_MONITOR,
            factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.REDUCE_LR_PATIENCE,
            min_lr=cfg.REDUCE_LR_MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=cfg.CHECKPOINT_MONITOR,
            save_best_only=True,
            verbose=1,
        ),
    ]


def _build_callbacks_v2(checkpoint_path: Path) -> list:
    cfg = CNNV2TrainingConfig
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor=cfg.EARLY_STOPPING_MONITOR,
            mode=cfg.EARLY_STOPPING_MODE,
            patience=cfg.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.REDUCE_LR_MONITOR,
            mode=cfg.REDUCE_LR_MODE,
            factor=cfg.REDUCE_LR_FACTOR,
            patience=cfg.REDUCE_LR_PATIENCE,
            min_lr=cfg.REDUCE_LR_MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=cfg.CHECKPOINT_MONITOR,
            mode=cfg.CHECKPOINT_MODE,
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(
    version: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
    seed: int = RANDOM_SEED,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, Path, Path]:
    """Train CNN V1 or V2 exactly as in the original notebook.

    Returns (model, history, checkpoint_path, final_model_path).
    """
    set_seeds(seed)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
    if X_val.ndim == 3:
        X_val = X_val[..., np.newaxis]

    input_shape = X_train.shape[1:]
    model = build_model(version, input_shape)
    model.summary(print_fn=logger.info)

    if version == "v1":
        cfg = CNNV1TrainingConfig
        class_weight: Optional[dict] = compute_balanced_class_weights(y_train)
        callbacks = _build_callbacks_v1(model_dir / cfg.CHECKPOINT_FILENAME)
    elif version == "v2":
        cfg = CNNV2TrainingConfig
        class_weight = None  # not used for V2 in the original notebook
        callbacks = _build_callbacks_v2(model_dir / cfg.CHECKPOINT_FILENAME)
    else:
        raise ValueError(f"Unknown model version: {version}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=cfg.LOSS,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    if class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(**fit_kwargs)

    final_model_path = model_dir / cfg.FINAL_MODEL_FILENAME
    model.save(final_model_path)

    checkpoint_path = model_dir / cfg.CHECKPOINT_FILENAME
    return model, history, checkpoint_path, final_model_path
