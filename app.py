import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Potato Blight Classifier", page_icon="🥔", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/potato_blight_classifier.keras")

def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    prob_healthy = float(model.predict(arr, verbose=0)[0][0])
    prob_blight = 1.0 - prob_healthy
    label = "Healthy" if prob_healthy >= 0.5 else "Early Blight"
    return label, prob_healthy * 100, prob_blight * 100

model = load_model()
uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=300)
    label, healthy_pct, blight_pct = predict(model, img)
    st.write(f"**Prediction:** {label}")
    st.progress(int(healthy_pct), text=f"Healthy: {healthy_pct:.1f}%")
    st.progress(int(blight_pct), text=f"Early Blight: {blight_pct:.1f}%")