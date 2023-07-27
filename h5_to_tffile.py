import tensorflow as tf
from tensorflow.keras.models import load_model

def make_tffile(model_path, save_path):
    model = load_model(model_path)

    # Convert the model to TensorFlow Lite format (.tflite)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open(save_path, 'wb') as f:
        f.write(tflite_model)