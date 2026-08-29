"""
CNN model architectures for binary stuttering classification.

The original notebook contains two architectures, trained sequentially:

  build_cnn_v1(): the first CNN trained in the notebook. Three
  Conv2D+BatchNorm+MaxPool+Dropout blocks (32/64/128 filters, dropout
  0.25/0.30/0.35), GlobalAveragePooling2D, a 128-unit dense layer, and
  a sigmoid output. Checkpoint file: best_cnn_model.keras.

  build_cnn_v2(): a smaller, more strongly regularized architecture
  trained afterwards (16/32/64 filters with L2 kernel regularization,
  SpatialDropout2D, a 64-unit dense layer). This is the architecture
  referenced throughout the accompanying research paper -- its
  checkpoint is named cnn_v2_best.keras, and the paper's reported test
  metrics (Accuracy 91.54%, Precision 71.77%, Recall 96.35%,
  F1 82.26%, ROC-AUC 96.89%) come from this model.

Both architectures are preserved exactly as implemented; neither
layer configuration, activation, nor regularization value has been
changed. Input tensors are the (120, 313, 1) MFCC+delta+delta-delta
feature maps produced by src/features/feature_extraction.py.
"""

from typing import Tuple

import tensorflow as tf


def build_cnn_v1(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """First CNN architecture trained in the original notebook."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.30),

            # Block 3
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.35),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.50),

            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_v1",
    )
    return model


def build_cnn_v2(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """CNN V2 -- the architecture whose results are reported in the paper."""
    l2 = tf.keras.regularizers.l2(1e-4)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),

            # Block 1
            tf.keras.layers.Conv2D(
                16, (3, 3), padding="same", activation="relu", kernel_regularizer=l2
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.SpatialDropout2D(0.20),

            # Block 2
            tf.keras.layers.Conv2D(
                32, (3, 3), padding="same", activation="relu", kernel_regularizer=l2
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.SpatialDropout2D(0.25),

            # Block 3
            tf.keras.layers.Conv2D(
                64, (3, 3), padding="same", activation="relu", kernel_regularizer=l2
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.SpatialDropout2D(0.30),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2),
            tf.keras.layers.Dropout(0.50),

            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="cnn_v2",
    )
    return model


MODEL_BUILDERS = {
    "v1": build_cnn_v1,
    "v2": build_cnn_v2,
}


def build_model(version: str, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Build a model by version string ("v1" or "v2"). Defaults elsewhere
    in this project point to "v2" since that is the architecture whose
    results are reported in the paper.
    """
    if version not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model version '{version}'. Expected one of: "
            f"{sorted(MODEL_BUILDERS)}"
        )
    return MODEL_BUILDERS[version](input_shape)
