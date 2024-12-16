import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer

# Global variables
img_size = 224  # Image size for ResNet50 model
channels = 3  # RGB channels
tflite_model_path = ''  # Path to the converted TensorFlow Lite model
classes = ['helicopter', 'drone', 'plane']  # Update with your class names
label_binarizer = LabelBinarizer()
label_binarizer.fit(classes)  # Fit the label binarizer with your class labels

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize image pixels to [0, 1]
    return img_array

# Function to make inference using TensorFlow Lite model
def predict(image_path):
    # Preprocess the input image
    img_array = preprocess_image(image_path)

    # Get input and output tensors from the interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Decode the prediction
    predicted_class_index = np.argmax(output_data, axis=1)
    predicted_class = classes[predicted_class_index[0]]

    # Get the confidence of the prediction (probability)
    confidence = output_data[0][predicted_class_index[0]]

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")
    
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    image_path = ""

    # Check if the file exists
    if os.path.isfile(image_path):
        predicted_class, confidence = predict(image_path)
    else:
        print("The file does not exist. Please provide a valid file path.")
