import tensorflow as tf
import argparse
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from dataloader import make_dataset
from model import student_model

teacher_model = load_model('/content/drive/MyDrive/gym_classification/log/vitl16_new.h5')

# Setting hyperparameters for distillation
temperature = 5.0  
alpha = 0.6      

# Set the hyperparameters for the Adam optimizer
beta_1 = 0.9
beta_2 = 0.999

# Create an instance of the Adam optimizer with the specified hyperparameters
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=beta_1, beta_2=beta_2)


def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Training setting', add_help=False)
    parser.add_argument('--train_dir', type=str, help='root path of the train dataset')
    parser.add_argument('--val_dir', type=str, help='root path of the val dataset')
    parser.add_argument('--image_width', type=int, default=224, help='resize the image')
    parser.add_argument('--image_height', type=int, default=224, help='resize the image')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    parser.add_argument('--num_classes', type=int, default=8, help='number of the class')
    parser.add_argument('--epochs', type=int, default=30, help='set epoch')
    

    return parser


# Define a function to compute the knowledge distillation loss
def knowledge_distillation_loss(y_true, y_pred, teacher_logits, temperature):
    soft_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)) \
              + tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.nn.softmax(teacher_logits / temperature),
                                                                        tf.nn.softmax(y_pred / temperature)))


    hard_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True))

    return alpha * hard_loss + (1. - alpha) * soft_loss

# Training loop
def train_step(images, labels, teacher_model, student_model, optimizer):
    with tf.GradientTape() as tape:

        student_predictions = student_model(images, training=True)

        teacher_logits = teacher_model(images, training=False)


        loss = knowledge_distillation_loss(labels, student_predictions, teacher_logits, temperature)

    # Compute gradients and apply them
    gradients = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

    return loss


def validate_student(val_generator, student_model, batch_size):
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    for images, labels in val_generator:
        predictions = student_model(images, training=False)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, predictions))
        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))
        total_loss += loss
        total_accuracy += accuracy
        num_batches += 1
        if num_batches >= val_generator.samples // batch_size:  # Ensure we iterate only once over the dataset
            break
    return total_loss / num_batches, total_accuracy / num_batches



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training model', parents=[get_args_parser()])
    args = parser.parse_args()
    train_generator, val_generator = make_dataset(args.train_dir, args.val_dir, args.image_width, args.image_height, args.batch_size)
    
    # Training iterations
    best_val_accuracy = 0.0  # To keep track of the best validation accuracy

    epochs = 45
    for epoch in range(epochs):
        total_train_loss = 0.0
        num_train_batches = 0
        for images, labels in train_generator:
            train_loss = train_step(images, labels, teacher_model, student_model, optimizer)
            total_train_loss += train_loss
            num_train_batches += 1
            if num_train_batches >= train_generator.samples // args.batch_size:  # Ensure we iterate only once over the dataset
                break

        val_loss, val_accuracy = validate_student(val_generator, student_model, args.batch_size)

        # Save the model if the validation accuracy is the best seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            student_model.save('/content/drive/MyDrive/gym_classification/log/best_student_model_9.h5')
            print(f"Best model saved with accuracy: {best_val_accuracy}")

        print(f"Epoch {epoch + 1}, Train Loss: {total_train_loss / num_train_batches}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    student_model.save('/content/drive/MyDrive/gym_classification/log/student_model_9.h5')
