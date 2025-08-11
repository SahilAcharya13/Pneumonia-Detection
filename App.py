import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('pneumonia_cnn_model.h5')

# Title
st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image, and the model will predict whether it's NORMAL or PNEUMONIA.")

# Image uploader
uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Chest X-ray', use_column_width=True)

    # Preprocess the image
    img = img.resize((180, 180))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    percentage = prediction * 100
    normal_prob = 100 - percentage

    # Display result
    if prediction < 0.5:
        st.success(f"✅ **NORMAL** - Confidence: {normal_prob:.2f}%")
    else:
        st.error(f"⚠️ **PNEUMONIA** Detected - Confidence: {percentage:.2f}%")
