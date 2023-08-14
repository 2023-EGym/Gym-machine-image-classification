import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from dataloader import make_dataset
#from model import model_2
from model import teacher_model

def get_args_parser():
    # define the argparse for the script
    parser = argparse.ArgumentParser('Training setting', add_help=False)
    parser.add_argument('--train_dir', type=str, help='root path of the train dataset')
    parser.add_argument('--val_dir', type=str, help='root path of the val dataset')
    parser.add_argument('--save_path', type=str, help='save best model')
    parser.add_argument('--image_width', type=int, default=224, help='resize the image')
    parser.add_argument('--image_height', type=int, default=224, help='resize the image')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--num_classes', type=int, default=7, help='number of the class')
    parser.add_argument('--epochs', type=int, default=7, help='set epoch')
    

    return parser

def check_point(save_path):
    checkpointer = ModelCheckpoint(
        save_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
        )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                                    factor = 0.2,
                                                    patience = 2,
                                                    verbose = 1,
                                                    min_delta = 1e-4,
                                                    min_lr = 1e-6,
                                                    mode = 'max')

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                                                    min_delta = 1e-4,
                                                    patience = 5,
                                                    mode = 'max',
                                                    restore_best_weights = True,
                                                    verbose = 1)

    callbacks = [earlystopping, reduce_lr, checkpointer]
    
    return callbacks


def main(train_generator, val_generator, model, callbacks, batch_size, epochs):
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=callbacks
    )

    # Save the trained model (optional)
    model.save('/content/drive/MyDrive/gym_classification/log/add_data_model2_30ep.h5')



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training model', parents=[get_args_parser()])
    args = parser.parse_args()
    train_generator, val_generator = make_dataset(args.train_dir, args.val_dir, args.image_width, args.image_height, args.batch_size)
    model = teacher_model(args.num_classes, input_shape = (args.image_width, args.image_height, 3))
    callbacks = check_point(args.save_path)
    main(train_generator, val_generator, model, callbacks, args.batch_size, args.epochs)
