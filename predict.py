import json
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_PATH = "models/class_names.json"

def predict(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(probs))
    print(f"Prediction: {classes[idx]}")
    print(f"Confidence: {probs[idx] * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/leaf.jpg")
        raise SystemExit(1)
    predict(sys.argv[1])
