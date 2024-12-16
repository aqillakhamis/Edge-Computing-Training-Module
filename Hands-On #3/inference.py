import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

# Global variables
img_size = 224  # Image size for ResNet50 model
channels = 3  # RGB channels
model_path = 'resnet50.h5'  # Path to the saved model
classes = ['helicopter', 'drone', 'plane']  # Update with your class names
label_binarizer = LabelBinarizer()
label_binarizer.fit(classes)  # You need to fit the label binarizer with your class labels

# Load the trained model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image pixels to [0, 1]
    return img_array

# Function to make inference
def predict(image_path):
    # Preprocess the input image
    img_array = preprocess_image(image_path)

    # Make the prediction
    predictions = model.predict(img_array)

    # Decode the prediction
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class = classes[predicted_class_index[0]]

    # Get the confidence of the prediction (probability)
    confidence = predictions[0][predicted_class_index[0]]

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    image_path = "dataset/drone/71JE-QG1FiL.jpg"

    # Check if the file exists
    if os.path.isfile(image_path):
        predicted_class, confidence = predict(image_path)
    else:
        print("The file does not exist. Please provide a valid file path.")
