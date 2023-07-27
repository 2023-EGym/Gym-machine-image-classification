import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Training setting', add_help=False)
    parser.add_argument('--save_path', type=str, help='save path')
    parser.add_argument('--model_path', type=str, help='root path of saved model')


def make_tffile(model_path, save_path):
    model = load_model(model_path)

    # Convert the model to TensorFlow Lite format (.tflite)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open(save_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('make tffile....', parents=[get_args_parser()])
    args = parser.parse_args()

    make_tffile(args.model_path, args.save_path)
