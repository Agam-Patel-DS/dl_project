import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the pre-trained VGG16 model
model = load_model('artifacts/training/model.h5')  # Ensure the model file is in the same directory as this script

# Title and Description
st.title("Kidney Disease Prediction")
st.write("Upload a kidney medical image, and the app will predict whether it indicates kidney disease or not.")

# File Upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))  # Resize for VGG16
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Predict using the model
    prediction = model.predict(img_array)
    result = "Positive for Kidney Disease" if prediction[0][0] > 0.5 else "Negative for Kidney Disease"

    # Display Prediction
    st.write("Prediction Result:")
    st.success(result)
