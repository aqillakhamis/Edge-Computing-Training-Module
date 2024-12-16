import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model('resnet50.h5')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('resnet50.tflite', 'wb') as f:
    f.write(tflite_model)
