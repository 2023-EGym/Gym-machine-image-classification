import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def make_dataset(train_dir, val_dir, image_width, image_height, batch_size):
    train_data_dir = train_dir
    val_data_dir = val_dir
    img_height, img_width = image_height, image_width
    batch_size = batch_size

    # Use ImageDataGenerator to augment and preprocess the images
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        rescale=1.0 / 255
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator