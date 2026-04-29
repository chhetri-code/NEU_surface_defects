"""
utils/image_utils.py
────────────────────
PIL / NumPy helpers for image loading, validation, and preprocessing.
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image

from config import IMG_HEIGHT, IMG_WIDTH


# ── Supported types ───────────────────────────────────────────────────────────

SUPPORTED_FORMATS: tuple[str, ...] = ("jpg", "jpeg", "png", "bmp", "webp")


# ── Core helpers ──────────────────────────────────────────────────────────────

def bytes_to_pil(raw: bytes) -> Image.Image:
    """Convert raw bytes (from st.file_uploader or st.camera_input) to PIL."""
    return Image.open(io.BytesIO(raw))


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Prepare a PIL image for model inference:
      1. Convert to RGB (handles grayscale NEU images and RGBA screenshots).
      2. Resize to (IMG_HEIGHT, IMG_WIDTH) using high-quality Lanczos resampling.
      3. Cast to float32 array with shape (IMG_HEIGHT, IMG_WIDTH, 3).

    Pixel values remain in [0, 255]; the model's internal
    ImageNetNormalization layer handles the final rescaling.
    """
    img_rgb   = pil_img.convert("RGB")
    img_resized = img_rgb.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    return np.array(img_resized, dtype=np.float32)


def display_image(pil_img: Image.Image, max_display_px: int = 400) -> Image.Image:
    """
    Returns a display-sized version of the image (width ≤ max_display_px),
    preserving aspect ratio – used only for Streamlit rendering, not for
    model input.
    """
    w, h = pil_img.size
    if w > max_display_px:
        ratio = max_display_px / w
        pil_img = pil_img.resize((max_display_px, int(h * ratio)), Image.LANCZOS)
    return pil_img
