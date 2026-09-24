# NEU Surface Defect Classifier — Streamlit App

A production-ready Streamlit app that classifies steel surface defects
from the [NEU Surface Defect Dataset](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)
into **6 categories**:

| Class | Description |
|---|---|
| Crazing | Network of fine cracks |
| Inclusion | Foreign material embedded in steel |
| Patches | Localised rough/discoloured patches |
| Pitted Surface | Small pits from corrosion or impact |
| Rolled-in Scale | Oxidised scale pressed in during rolling |
| Scratches | Linear marks from abrasion |

---

## Project Structure

```
neu_defect_classifier/
├── app.py                 # Streamlit entry point
├── config.py              
├── requirements.txt
├── model/
│   ├── __init__.py
│   ├── builder.py         # Re-creates the model architecture
│   └── predictor.py       # Model loading & inference
└── utils/
    ├── __init__.py
    └── image_utils.py     # PIL / NumPy preprocessing helpers
```

---

## Step 1: Trained model from experimental notebook

```python
model.save("neu_defect_model.keras")
```

---

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Usage

The app provides two input methods:

- **📁 Upload Image** : drag-and-drop or browse for a `.jpg`, `.png`, `.bmp`, or `.webp` file.
- **📷 Capture from Camera** : take a live photo directly.

Once an image is in, the app:
1. Resizes it to 200 × 200 px (training resolution).
2. Converts it to RGB.
3. Runs MobileNetV2 inference (normalisation is applied inside the model).
4. Displays the **top predicted class** with its confidence score, plus a
   ranked bar chart of the top-3 predictions.

---

## Configuration

All tuneable constants in `config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `IMG_HEIGHT` / `IMG_WIDTH` | 200 | Model input resolution |
| `CLASS_NAMES` | (6 classes) | Label list in alphabetical/folder order |
| `MODEL_PATH` | `neu_defect_model.keras` | Path to the saved model |
| `TOP_K` | 3 | Predictions to display |

---

## Notes

- The model's `ImageNetNormalization` layer is a custom Keras layer registered
  with `@tf.keras.utils.register_keras_serializable`, so it survives
  serialisation/deserialisation without extra `custom_objects` machinery.
- The model is loaded once and cached with `@st.cache_resource`, so repeated
  image submissions don't reload it.
- Camera capture uses Streamlit's built-in `st.camera_input` widget — no extra
  browser permissions or libraries needed.
