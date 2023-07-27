import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Training setting', add_help=False)
    parser.add_argument('--test_dir', type=str, help='root path of the test dataset')
    parser.add_argument('--model_path', type=str, help='root path of saved model')


def inference(model_path, test_dir):
    model = load_model(model_path)

    # Prepare the test dataset
    test_data_dir = test_dir
    img_height, img_width = 224, 224
    batch_size = 32

    # Use ImageDataGenerator to preprocess the test images
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Make sure to set shuffle to False for inference
    )

    # Run inference on the test dataset
    predictions = model.predict(test_generator)

    # Get the predicted class labels (indices with the highest probability)
    predicted_labels = tf.argmax(predictions, axis=1).numpy()

    # Get the true class labels (ground truth) from the generator
    true_labels = test_generator.classes

    # Calculate accuracy for each class label
    num_classes = len(test_generator.class_indices)
    class_labels = list(test_generator.class_indices.keys())
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(true_labels)):
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        class_correct[true_label] += int(true_label == predicted_label)
        class_total[true_label] += 1

    # Print accuracy for each class label
    for i in range(num_classes):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"Class: {class_labels[i]} / Accuracy: {accuracy:.2f} / Correct: {class_correct[i]} / Total: {class_total[i]}")

    # Calculate overall accuracy on the test dataset
    accuracy = np.mean(predicted_labels == true_labels)
    print("Overall Test Accuracy:", accuracy)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Inference', parents=[get_args_parser()])
    args = parser.parse_args()

    inference(args.model_path, args.test_dir)
