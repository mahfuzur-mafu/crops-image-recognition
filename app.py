import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image


model = load_model('cropmodel.keras')


class_names = {
    0: 'Cherry', 1: 'Coffee-plant', 2: 'Cucumber', 3: 'Fox_nut(Makhana)', 4: 'Lemon', 
    5: 'Olive-tree', 6: 'Pearl_millet(bajra)', 7: 'Tobacco-plant', 8: 'almond', 9: 'banana', 
    10: 'cardamom', 11: 'chilli', 12: 'clove', 13: 'coconut', 14: 'cotton', 15: 'gram', 
    16: 'jowar', 17: 'jute', 18: 'maize', 19: 'mustard-oil', 20: 'papaya', 21: 'pineapple', 
    22: 'rice', 23: 'soyabean', 24: 'sugarcane', 25: 'sunflower', 26: 'tea', 
    27: 'tomato', 28: 'vigna-radiati(Mung)', 29: 'wheat'
}


st.title("🌿 Crops Image Recognition App")
st.write("Upload an image of a plant to recognize its category.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_resized, (224, 224))  


    st.image(image_resized, caption="Uploaded Image", use_container_width=True)


    image_input = np.expand_dims(image_resized, axis=0)
    image_input = preprocess_input(image_input)


    if st.button("Classify Image"):

        predictions = model.predict(image_input)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions)


        predicted_class_name = class_names[class_index]

        st.success(f"🌱 Predicted: **{predicted_class_name}** with **{confidence * 100:.2f}%** confidence")
