from flask import Flask, Response, render_template
import cv2
import tensorflow as tf
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained MobileNet SSD model
model = tf.saved_model.load('saved_model')  # Path to pre-trained model
category_index = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 72: "tv", 77: "cell phone"
    # Extend this as needed with COCO labels
}

# Initialize camera
camera = cv2.VideoCapture(0)

# Helper function to preprocess image
def preprocess_image(frame):
    # Resize frame to match the model input
    frame = cv2.resize(frame, (300, 300))
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
    input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension
    return input_tensor

# Helper function to detect objects and annotate frame
def detect_and_annotate(frame):
    input_tensor = preprocess_image(frame)
    detections = model(input_tensor)  # Run detection

    # Extract detection results
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape

    # Iterate through detections
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:  # Minimum confidence threshold
            class_id = detection_classes[i]
            label = category_index.get(class_id, "Unknown")
            box = detection_boxes[i]

            # Scale box to original frame dimensions
            y_min, x_min, y_max, x_max = (box[0] * height, box[1] * width,
                                          box[2] * height, box[3] * width)
            # Draw bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), 
                          (int(x_max), int(y_max)), (0, 255, 0), 2)
            # Add label and confidence
            label_text = f"{label}: {detection_scores[i]:.2f}"
            cv2.putText(frame, label_text, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    return frame

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Process the frame for object detection
                frame = detect_and_annotate(frame)

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield frame as part of a multipart HTTP response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
