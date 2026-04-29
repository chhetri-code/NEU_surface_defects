"""
app.py  –  NEU Surface Defect Classifier
─────────────────────────────────────────
Run with:
    streamlit run app.py

Expects `neu_defect_model.keras` in the same directory.
Save it from the training notebook with:
    model.save("neu_defect_model.keras")
"""

import streamlit as st

from config import APP_TITLE, APP_SUBTITLE, TOP_K
from model.predictor import load_model, predict
from utils.image_utils import bytes_to_pil, preprocess_image, display_image


# ── Page setup (must be the very first Streamlit call) ────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Colour map for each defect class ─────────────────────────────────────────
CLASS_COLOURS: dict[str, str] = {
    "Crazing":         "#e74c3c",
    "Inclusion":       "#e67e22",
    "Patches":         "#f1c40f",
    "Pitted Surface":  "#2ecc71",
    "Rolled-in Scale": "#3498db",
    "Scratches":       "#9b59b6",
}


# ═════════════════════════════════════════════════════════════════════════════
# 1. HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title(f"🔬 {APP_TITLE}")
st.caption(APP_SUBTITLE)
st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL LOADING  (cached – only runs once per session)
# ═════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading model…"):
    model = load_model()


# ═════════════════════════════════════════════════════════════════════════════
# 3. IMAGE ACQUISITION
# ═════════════════════════════════════════════════════════════════════════════
tab_upload, tab_camera = st.tabs(["📁  Upload Image", "📷  Capture from Camera"])

raw_image_bytes: bytes | None = None

with tab_upload:
    uploaded = st.file_uploader(
        "Choose a steel-surface image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        raw_image_bytes = uploaded.read()

with tab_camera:
    camera_snap = st.camera_input("Take a photo")
    if camera_snap:
        raw_image_bytes = camera_snap.read()


# ═════════════════════════════════════════════════════════════════════════════
# 4. INFERENCE & RESULTS  (only when an image is available)
# ═════════════════════════════════════════════════════════════════════════════
if raw_image_bytes is None or model is None:
    # No image yet – show a friendly prompt and fall through to footer
    if raw_image_bytes is None:
        st.info(
            "👆  Upload a steel surface image **or** take a photo with your "
            "camera to classify the defect.",
            icon="ℹ️",
        )

else:
    pil_img = bytes_to_pil(raw_image_bytes)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Input Image")
        st.image(display_image(pil_img), use_container_width=True)

    with st.spinner("Running inference…"):
        img_array = preprocess_image(pil_img)
        top_preds  = predict(model, img_array)

    top_class      = top_preds[0]["class"]
    top_confidence = top_preds[0]["confidence"]

    with col_result:
        st.subheader("Prediction")

        bar_colour = CLASS_COLOURS.get(top_class, "#636e72")

        # Primary result card
        st.markdown(
            f"""
            <div style="
                background: {bar_colour}22;
                border-left: 5px solid {bar_colour};
                border-radius: 6px;
                padding: 16px 20px;
                margin-bottom: 12px;
            ">
                <div style="font-size: 0.75rem; text-transform: uppercase;
                            letter-spacing: 1px; color: #636e72; margin-bottom: 4px;">
                    Detected Defect
                </div>
                <div style="font-size: 1.6rem; font-weight: 700; color: {bar_colour};">
                    {top_class}
                </div>
                <div style="font-size: 1.05rem; color: #2d3436; margin-top: 4px;">
                    Confidence: <b>{top_confidence:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top-K confidence bars
        st.markdown(f"**Top {TOP_K} predictions**")
        for rank, pred in enumerate(top_preds):
            cls    = pred["class"]
            conf   = pred["confidence"]
            colour = CLASS_COLOURS.get(cls, "#636e72")
            label  = f"{'🥇' if rank == 0 else '·'} {cls}"
            st.markdown(
                f"""
                <div style="margin-bottom: 6px;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.85rem; margin-bottom:2px;">
                        <span>{label}</span>
                        <span><b>{conf:.1f}%</b></span>
                    </div>
                    <div style="background:#ecf0f1; border-radius:4px; height:8px;">
                        <div style="
                            width:{conf:.1f}%;
                            background:{colour};
                            border-radius:4px;
                            height:8px;">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Class legend (shown after results)
    with st.expander("ℹ️  About the defect classes"):
        descriptions = {
            "Crazing":         "Network of fine cracks on the surface.",
            "Inclusion":       "Foreign material embedded in the steel surface.",
            "Patches":         "Localised rough or discoloured patches.",
            "Pitted Surface":  "Small pits or holes caused by corrosion or impact.",
            "Rolled-in Scale": "Oxidised scale pressed into the surface during rolling.",
            "Scratches":       "Linear marks from abrasion or mechanical contact.",
        }
        for cls, desc in descriptions.items():
            dot = f"<span style='color:{CLASS_COLOURS[cls]};font-size:1.1rem;'>●</span>"
            st.markdown(f"{dot} &nbsp; **{cls}** — {desc}", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. FOOTER  (always visible)
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Model: MobileNetV2 fine-tuned on the "
    "[NEU Surface Defect Dataset](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)"
    "  |  Framework: TensorFlow / Keras"
)
st.caption("Made in 🇮🇳 with ❤️ by CHHETRI")
