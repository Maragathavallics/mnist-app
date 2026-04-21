import streamlit as st
from keras.models import load_model
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import matplotlib.pyplot as plt

model = load_model('mnist.keras')

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0–9) and click Predict")

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

if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0]
        img = Image.fromarray(img.astype('uint8'))
        img = img.resize((28,28)).convert('L')
        
        img = np.array(img)
        img = img.reshape(1,28,28,1)
        img = img / 255.0
        
        pred = model.predict(img)
        probs = pred[0]
        digit = np.argmax(probs)
        confidence = np.max(probs)

        st.success(f"Prediction: {digit}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        # Top 3 predictions
        st.write("Top 3 Predictions:")
        top3 = probs.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"{i}: {probs[i]*100:.2f}%")

        # Probability chart
        fig, ax = plt.subplots()
        ax.bar(range(10), probs)
        ax.set_xlabel("Digits")
        ax.set_ylabel("Probability")
        st.pyplot(fig)