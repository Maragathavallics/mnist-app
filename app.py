import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt

# 🔧 Page config (UI improvement)
st.set_page_config(page_title="Digit Recognizer", layout="centered")

# 🔧 Cache model (prevents reload issues)
import requests
import os

@st.cache_resource
def load_my_model():
    model_path = "mnist.h5"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?export=download&id=1eoIQ0JDCqw5VtnXzRauewv2aefFahACf"
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)

    return load_model(model_path, compile=False)

model = load_my_model()

# 🧠 Title
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click Predict")

# 🎨 Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 🔥 Clear button (for demo)
if st.button("Clear"):
    st.experimental_rerun()

# 🔍 Prediction
if st.button("Predict"):
    # 🔧 Handle empty canvas
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:

        # 🖼️ Preprocess image
        img = canvas_result.image_data[:, :, 0]
        img = 255 - img  # 🔧 Invert image for MNIST

        img = Image.fromarray(img.astype("uint8"))
        img = img.resize((28, 28)).convert("L")

        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0

        # 🤖 Prediction
        pred = model.predict(img)
        probs = pred[0]
        digit = np.argmax(probs)
        confidence = np.max(probs)

        # 🎯 Output
        st.success(f"Prediction: {digit}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        # 🔝 Top 3 predictions
        st.write("Top 3 Predictions:")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"{i}: {probs[i]*100:.2f}%")

        # 📊 Probability chart
        fig, ax = plt.subplots()
        ax.bar(range(10), probs)
        ax.set_xlabel("Digits")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)

    else:
        st.warning("⚠️ Please draw a digit first!")
