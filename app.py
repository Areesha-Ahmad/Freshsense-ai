"""FreshSense AI - Streamlit demo: upload a fruit photo, get fresh/rotten verdict."""
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).parent
IMG = 48


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BASE / "freshsense_ann.keras")


def predict(model, pil_img):
    img = pil_img.convert("RGB").resize((IMG, IMG))
    x = np.asarray(img, dtype=np.float32)[None] / 255.0
    p_rotten = float(model.predict(x, verbose=0).ravel()[0])
    return p_rotten


st.set_page_config(page_title="FreshSense AI", page_icon="🍎", layout="centered")
st.title("🍎 FreshSense AI")
st.write("AI food-freshness detector — upload a photo of a fruit and the ANN "
         "tells you if it looks **fresh** or **rotten**.")

model = load_model()
up = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png", "webp"])

if up is not None:
    pil = Image.open(up)
    st.image(pil, caption="Uploaded image", use_container_width=True)
    with st.spinner("Analysing..."):
        p_rotten = predict(model, pil)
    p_fresh = 1 - p_rotten
    if p_rotten >= 0.5:
        st.error(f"🤢 **ROTTEN** — confidence {p_rotten:.1%}")
    else:
        st.success(f"✅ **FRESH** — confidence {p_fresh:.1%}")
    st.progress(p_fresh, text=f"Freshness meter: {p_fresh:.0%} fresh")
    with st.expander("See model probabilities"):
        st.write(f"Fresh: {p_fresh:.1%} | Rotten: {p_rotten:.1%}")
else:
    st.info("👆 Choose an image to get started.")

st.caption("Simple dense ANN trained on the fresh/rotten fruit dataset. "
           "Demo only — not a food-safety device.")
