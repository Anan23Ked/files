# ─────────────────────────────────────────────────────────────────
#  model.py  —  CNN using tf.keras layers (stable BatchNorm)
#               but trained with a pure TF GradientTape loop
# ─────────────────────────────────────────────────────────────────

import tensorflow as tf
import numpy as np
from config import IMG_SIZE, CHANNELS, NUM_CLASSES


class ExpressionCNN:
    """
    CNN for facial expression classification.
    Uses tf.keras layers internally for stable BatchNorm behaviour,
    but exposes a raw forward() method for GradientTape training.

    Architecture
    ─────────────
    Input  (48 x 48 x 1  grayscale)
      Conv2D(32)  + BN + ReLU + MaxPool + Dropout(0.25)
      Conv2D(64)  + BN + ReLU + MaxPool + Dropout(0.25)
      Conv2D(128) + BN + ReLU + MaxPool + Dropout(0.25)
      Flatten
      Dense(256)  + BN + ReLU + Dropout(0.5)
      Dense(num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self._build()

    def _build(self):
        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))

        # Block 1
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', use_bias=False)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        # Block 2
        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        # Block 3
        x = tf.keras.layers.Conv2D(256, (3,3), padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2,2)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        # Classifier head
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(self.num_classes)(x)

        self._model = tf.keras.Model(inputs, outputs, name='ExpressionCNN')

    def forward(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self._model(x, training=training)

    def trainable_variables(self) -> list:
        return self._model.trainable_variables

    def save_weights(self, path: str):
        h5_path = path.replace('.npy', '.weights.h5')
        self._model.save_weights(h5_path)
        print(f"[Model] Weights saved -> {h5_path}")

    def load_weights(self, path: str):
        h5_path = path.replace('.npy', '.weights.h5')
        self._model.load_weights(h5_path)
        print(f"[Model] Weights loaded <- {h5_path}")

    def summary(self):
        self._model.summary()
        print(f"\n[Model] Total params : {self._model.count_params():,}")
        print(f"[Model] Classes      : {self.num_classes}")
        print(f"[Model] Input shape  : ({IMG_SIZE} x {IMG_SIZE} x {CHANNELS})\n")