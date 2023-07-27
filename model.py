import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def model_1(num_classes, input_shape = (224, 224, 3)):
    # Load the pre-trained ResNet50 model (without the top classification layers)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new classifier on top (unfrozen)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of classes in your custom dataset
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


def model_2(num_classes, input_shape = (224, 224, 3)):
    # Load the pre-trained ResNet50 model (without the top classification layers)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers except for the last convolutional block (conv4)
    for layer in base_model.layers[:-36]:  # Fine-tune the last 36 layers (conv4 and beyond)
        layer.trainable = False

    # Add a new FC layer on top (unfrozen)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of classes in your custom dataset

    # Create the model with the new FC layer
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model


def model_3(num_classes, input_shape = (224, 224, 3)):
    # Load the pre-trained ResNet50 model (without the top classification layers)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new FC layers on top
    num_ftrs = base_model.output_shape[1]  # Get the number of features in the output of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # num_classes is the number of classes in your custom dataset
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze all layers except for the new FC layers
    for layer in model.layers[:-3]:  # Exclude the last 3 layers (the new FC layers)
        layer.trainable = False
        
    return model 