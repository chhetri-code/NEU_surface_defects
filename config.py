# ──────────────────────────────────────────────────────────────────────────────
# config.py  –  Central configuration for the NEU Surface Defect Classifier app
# ──────────────────────────────────────────────────────────────────────────────

# --------------- Model input dimensions (must match training) -----------------
IMG_HEIGHT: int = 200
IMG_WIDTH:  int = 200

# --------------- Defect categories (folder-name order from ImageFolder) -------
CLASS_NAMES: list[str] = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches",
]
NUM_CLASSES: int = len(CLASS_NAMES)

# --------------- Path to the saved Keras model --------------------------------
# Export from the notebook with:
#     model.save("neu_defect_model.keras")
# then copy the file next to this project.
MODEL_PATH: str = "neu_defect_model.keras"

# --------------- ImageNet normalisation statistics ----------------------------
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# --------------- UI / display -------------------------------------------------
APP_TITLE:    str = "NEU Surface Defect Classifier using MobilenetV2 with Custom Head"
APP_SUBTITLE: str = "Upload or capture a steel-surface image"
TOP_K:        int = 3          # how many top predictions to show
