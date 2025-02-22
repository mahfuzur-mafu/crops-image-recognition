# Image Classification for Agricultural Crops

## Project Overview
This project utilizes deep learning to classify images of various agricultural crops. It leverages the ResNet50 architecture as a pre-trained feature extractor and builds a custom classifier on top to identify 30 different crop types.

## Key Features
- **Image Preprocessing**: Uses `ImageDataGenerator` for image augmentation, including rotation, shifting, shearing, zooming, and flipping, to improve model generalization. Also applies `preprocess_input` from `keras.applications.resnet50` for image preprocessing.
- **Data Splitting**: The dataset is split into training, validation, and testing sets using the `splitfolders` library.
- **Transfer Learning**: Employs transfer learning with the ResNet50 model, pre-trained on ImageNet, to extract powerful image features.
- **Custom Classifier**: Builds a sequential model with global average pooling, dense layers, and dropout for classification.
- **Model Training**: Trains the model on the augmented training data, with validation performed during training.
- **Model Evaluation**: Evaluates the trained model's accuracy on an unseen test dataset.
- **Image Prediction**: Provides a function to predict the class of a single image, given the trained model.
- **Model Saving**: Saves the trained model as `cropmodel.keras`.

## Libraries Used
- TensorFlow (with Keras)
- NumPy
- Matplotlib
- OpenCV (cv2)
- Pillow
- splitfolders

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

## Possible Improvements
- Enhance data augmentation techniques.
- Further optimize hyperparameters for improved performance.
- Experiment with alternative deep learning architectures like EfficientNet.

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




