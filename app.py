import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Satellite Image Classifier", layout="centered")

# Title
st.title("🌍 Satellite Image Classification")
st.write("Upload an image and let the model classify it as Cloudy, Desert, Green Area, or Water.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Modelenv.v1.h5")
    return model

model = load_model()

# Class names as per your dataset
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Image upload
uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((256, 256))  # Resize to match model input
    img_array = np.array(img)

    if img_array.shape[-1] == 4:  # Handle PNG with alpha channel
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    st.success(f"✅ Predicted: **{predicted_label}**")
    st.write("📊 Confidence Scores:")
    for name, score in zip(class_names, predictions[0]):
        st.write(f"- {name}: {score:.2%}")
