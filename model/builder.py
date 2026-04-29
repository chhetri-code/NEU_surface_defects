"""
model/builder.py
────────────────
Re-creates the exact same model architecture used during training so that
saved weights can be loaded reliably, even when the .keras file contains
custom layers.
"""

import tensorflow as tf
from config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ── Custom preprocessing layer (mirrors the notebook's ImageNetNormalization) ─

@tf.keras.utils.register_keras_serializable(package="NEUApp")
class ImageNetNormalization(tf.keras.layers.Layer):
    """
    Converts uint8-like [0-255] images to float32 and applies
    channel-wise ImageNet mean/std normalisation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
        self.std  = tf.constant(IMAGENET_STD,  dtype=tf.float32)

    def call(self, inputs):                          # noqa: D102
        x = tf.cast(inputs, tf.float32) / 255.0
        return (x - self.mean) / self.std

    def get_config(self):                            # noqa: D102
        return super().get_config()


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model() -> tf.keras.Model:
    """
    Returns a compiled MobileNetV2-based transfer-learning model identical
    to the one created in the training notebook (feature-extraction mode,
    backbone frozen).
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ],
        name="augmentation_layer",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs  = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x       = data_augmentation(inputs)
    x       = ImageNetNormalization()(x)
    x       = base_model(x, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="Defect_Classifier_Feature_Extractor")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
