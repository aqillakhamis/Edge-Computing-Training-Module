import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels file
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

print(labels)

# load the labels by define in the inference script
labels = ['0 Helicopter', '1 Drone', '2 Plane']

print(labels)


# Preprocess the image for grayscale and the correct size
def preprocess_image(frame, input_size):
    # Convert the frame to grayscale if it's not already
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to match the input size
    resized_frame = cv2.resize(gray_frame, input_size)

    # Normalize pixel values to [0, 1]
    normalized_frame = resized_frame.astype(np.float32) / 255.0

    # Add the batch dimension and channel dimension (1 channel for grayscale)
    input_data = np.expand_dims(normalized_frame, axis=-1)  # Now shape should be (96, 96, 1)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension (1, 96, 96, 1)
    
    return input_data

# Perform inference
def classify_frame(frame):
    input_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])  # 96, 96
    input_data = preprocess_image(frame, input_size)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output results
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(output_data)
    predicted_label = labels[predicted_index]
    confidence = output_data[predicted_index]
    
    return predicted_label, confidence

# Initialize the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Starting real-time classification. Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform classification on the frame
    label, confidence = classify_frame(frame)

    # Display the results on the frame
    cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Real-Time Classification", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()