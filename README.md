## Project Overview
This project utilizes deep learning to classify images of various agricultural crops. It leverages the ResNet50 architecture as a pre-trained feature extractor and builds a custom classifier on top to identify 30 different crop types. Additionally, a Streamlit-based web application allows users to upload images and receive crop predictions.

## Key Features
- **Image Preprocessing**: Uses `ImageDataGenerator` for image augmentation, including rotation, shifting, shearing, zooming, and flipping, to improve model generalization. Also applies `preprocess_input` from `keras.applications.resnet50` for image preprocessing.
- **Data Splitting**: The dataset is split into training, validation, and testing sets using the `splitfolders` library.
- **Transfer Learning**: Employs transfer learning with the ResNet50 model, pre-trained on ImageNet, to extract powerful image features.
- **Custom Classifier**: Builds a sequential model with global average pooling, dense layers, and dropout for classification.
- **Model Training**: Trains the model on the augmented training data, with validation performed during training.
- **Model Evaluation**: Evaluates the trained model's accuracy on an unseen test dataset.
- **Image Prediction**: Provides a function to predict the class of a single image, given the trained model.
- **Model Saving**: Saves the trained model as `cropmodel.keras`.
- **Web App for Image Classification**: A Streamlit-based web interface allows users to upload images and classify them using the trained model.

## Libraries Used
- TensorFlow (with Keras)
- NumPy
- Matplotlib
- OpenCV (cv2)
- Pillow
- splitfolders
- Streamlit

## Dataset
The project was developed using a dataset of agricultural crops, divided into 30 classes. The images are split into training, validation, and testing sets.

## How It Works
1. The dataset is split into train, validation, and test folders.
2. Image augmentation is applied to the training data.
3. A ResNet50 model is loaded (pre-trained on ImageNet) and the final layers are replaced with new classification layers.
4. The model is trained with the training dataset, with validation performed during training.
5. The model is then evaluated with the test dataset.
6. The final model is saved as `cropmodel.keras`.
7. An image can be passed to the model for classification using the `predict_img` function.
8. The trained model is integrated into a Streamlit web app for easy image classification.

## Streamlit Web Application
To provide an interactive experience, the model is deployed as a web application using Streamlit.

### How to Use the Web App
1. Upload an image of a crop using the file uploader.
2. The image is resized and preprocessed.
3. Click on the "Classify Image" button to predict the crop type.
4. The predicted class and confidence score are displayed.


## Code Example (Predicting a New Image)
```python
# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('cropmodel.keras')

# Load the class names
class_names = {
    0: 'Cherry', 1: 'Coffee-plant', 2: 'Cucumber', 3: 'Fox_nut(Makhana)', 4: 'Lemon', 
    5: 'Olive-tree', 6: 'Pearl_millet(bajra)', 7: 'Tobacco-plant', 8: 'almond', 9: 'banana', 
    10: 'cardamom', 11: 'chilli', 12: 'clove', 13: 'coconut', 14: 'cotton', 15: 'gram', 
    16: 'jowar', 17: 'jute', 18: 'maize', 19: 'mustard-oil', 20: 'papaya', 21: 'pineapple', 
    22: 'rice', 23: 'soyabean', 24: 'sugarcane', 25: 'sunflower', 26: 'tea', 
    27: 'tomato', 28: 'vigna-radiati(Mung)', 29: 'wheat'
}

# Function to predict class of an image
def predict_img(image, model):
    import cv2
    import numpy as np
    test_img = cv2.imread(image)
    test_img = cv2.resize(test_img, (224, 224))
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    r = np.argmax(result)
    print(class_names[r])

# Predict a new image
predict_img('path/to/your/image.jpg', model)
```

## How to Run
1. Install the required libraries:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python Pillow splitfolders
   ```
2. Set up the dataset folders (`Agricultural-crops`).
3. Create a folder `ImageRecognition` in your Google Drive and move the dataset inside it.
4. Run the notebook to train the model.



### Streamlit Code Example
```python
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

# Load the trained model
model = load_model('cropmodel.keras')

# Class names mapping
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
```

## Possible Improvements
- Enhance data augmentation techniques.
- Further optimize hyperparameters for improved performance.
- Experiment with alternative deep learning architectures like EfficientNet.
- Deploy the web app using cloud platforms for accessibility.

## How to Run
1. Install the required libraries:
   ```bash
   pip install tensorflow numpy matplotlib opencv-python Pillow splitfolders streamlit
   ```
2. Set up the dataset folders (`Agricultural-crops`).
3. Create a folder `ImageRecognition` in your Google Drive and move the dataset inside it.
4. Train the model by running the notebook.
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

##Result

<img width="1467" alt="image" src="https://github.com/user-attachments/assets/7dac8596-526e-48d9-aec3-305f49830c40" />

<img width="1467" alt="image" src="https://github.com/user-attachments/assets/7dc0f382-ddd5-4397-bfd5-ffdd1cda9b17" />

<img width="1467" alt="image" src="https://github.com/user-attachments/assets/9eb5de2d-ce68-4331-9c36-d9c02f85ebd6" />




