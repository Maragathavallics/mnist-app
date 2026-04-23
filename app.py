import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown

# 🔧 Page config
st.set_page_config(page_title="Digit Recognizer", layout="centered")

# 🚀 Load model from Google Drive
@st.cache_resource
def load_my_model():
    model_path = "mnist.h5"

    if not os.path.exists(model_path):
        file_id = "1xCQtE8pXEaCQ4lvvn-hpK03NYMU4umZP"  # your drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)

    return load_model(model_path, compile=False)

model = load_my_model()

# 🎯 Title
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click Predict")

# 🎨 Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=8,   # thinner strokes for better accuracy
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 🔄 Clear button
if st.button("Clear"):
    st.rerun()

# 🔍 Prediction
if st.button("Predict"):

    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:

        # 🧠 Preprocessing (IMPROVED)
        img = canvas_result.image_data[:, :, 0]

        # invert colors
        img = 255 - img

        # remove noise
        img[img < 100] = 0

        # crop to digit (center it)
        coords = np.column_stack(np.where(img > 0))
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            img = img[y_min:y_max, x_min:x_max]

        # resize to 28x28
        img = Image.fromarray(img.astype("uint8"))
        img = img.resize((28, 28))

        # normalize
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        # 🤖 Prediction
        pred = model.predict(img)
        probs = pred[0]
        digit = np.argmax(probs)
        confidence = np.max(probs)

        # ✅ Output
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
