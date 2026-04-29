"""
model/predictor.py
──────────────────
Handles model loading (with caching) and single-image inference.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import streamlit as st

from config import MODEL_PATH, CLASS_NAMES, TOP_K
from model.builder import ImageNetNormalization   # ensures custom layer is registered


# ── Model loader (cached so the model is only loaded once per session) ────────

@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model | None:
    """
    Loads the saved Keras model.  Returns None if the file is missing so that
    the UI can show a friendly error instead of crashing.
    """
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"ImageNetNormalization": ImageNetNormalization},
        )
        return model
    except (OSError, ValueError) as exc:
        st.error(
            f"**Could not load model from** `{MODEL_PATH}`.\n\n"
            "Make sure you have:\n"
            "1. Trained the model in the notebook.\n"
            "2. Saved it with `model.save('neu_defect_model.keras')`.\n"
            "3. Placed `neu_defect_model.keras` in the same folder as `app.py`.\n\n"
            f"*(Error: {exc})*"
        )
        return None


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(model: tf.keras.Model, image_array: np.ndarray) -> list[dict]:
    """
    Run inference on a single pre-processed image.

    Parameters
    ----------
    model       : loaded Keras model
    image_array : float32 array of shape (IMG_HEIGHT, IMG_WIDTH, 3), pixel
                  values in [0, 255] (normalisation is inside the model).

    Returns
    -------
    List of dicts sorted by descending confidence, length == TOP_K:
        [{"class": str, "confidence": float (0-100)}, ...]
    """
    # Add batch dimension → (1, H, W, 3)
    batch = np.expand_dims(image_array, axis=0)

    probs = model.predict(batch, verbose=0)[0]          # shape: (NUM_CLASSES,)

    # Build full list, then sort and slice top-K
    results = [
        {"class": cls, "confidence": float(prob) * 100}
        for cls, prob in zip(CLASS_NAMES, probs)
    ]
    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results[:TOP_K]
