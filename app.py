import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="DeepVision", page_icon="🌿", layout="centered")

MODEL_PATH = Path("models/plant_disease_model.keras")
CLASS_PATH = Path("models/class_names.json")
IMG_SIZE = (224, 224)

st.title("🌿 DeepVision")
st.subheader("Plant Disease Detection using Deep Learning")
st.write("Upload a clear plant-leaf image and the trained CNN model will predict its class.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    return json.loads(CLASS_PATH.read_text(encoding="utf-8"))

if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.warning("Model not found. Run `python train.py` after adding the dataset.")
    st.stop()

model = load_model()
class_names = load_classes()

uploaded = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    arr = np.array(image.resize(IMG_SIZE), dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Analyzing leaf..."):
        probabilities = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(probabilities))
    confidence = float(probabilities[idx]) * 100

    st.success(f"Prediction: {class_names[idx]}")
    st.metric("Confidence", f"{confidence:.2f}%")

    top = np.argsort(probabilities)[::-1][:5]
    st.write("### Top predictions")
    for i in top:
        st.write(f"**{class_names[i]}** — {probabilities[i]*100:.2f}%")

st.info("This is an academic prototype. Predictions should not be treated as professional agricultural advice.")
