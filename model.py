import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from vit_keras import vit


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

def teacher_model(num_classes, input_shape = (224, 224, 3)):
    vit_model = vit.vit_l16(
        image_size=img_height,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
    
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = vit_model(inputs)
    embedding = layers.Dense(128)(x)
    
    classifier_output = layers.Dense(num_classes, activation="softmax")(embedding)
    
    model = Model(inputs, classifier_output)

    beta_1 = 0.9
    beta_2 = 0.999
    adam_optimizer = Adam(learning_rate=1e-4, beta_1=beta_1, beta_2=beta_2)
    
    model.compile(optimizer=adam_optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def student_model(num_classes, input_shape = (224, 224, 3)):
    # Create the student model using ResNet-50
    student_inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    student_resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    student_x = student_resnet_model(student_inputs)
    student_x = GlobalAveragePooling2D()(student_x)
    student_embedding = Dense(256, activation='relu', kernel_initializer='he_normal')(student_x)
    student_predictions = Dense(num_classes, activation="softmax")(student_embedding)
    student_model = Model(student_inputs, student_predictions)
    
    beta_1 = 0.9
    beta_2 = 0.999
    adam_optimizer = Adam(learning_rate=1e-4, beta_1=beta_1, beta_2=beta_2)
    
    student_model.compile(optimizer=adam_optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

    return student_model
