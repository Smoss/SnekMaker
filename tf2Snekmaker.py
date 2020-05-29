# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:25:28 2020

@author: smoss
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Reshape, Conv2DTranspose
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ProgbarLogger
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.activations import tanh

import os

from tqdm import tqdm

import numpy as np
from tensorflow.keras import optimizers
import argparse
import pandas as pd
import ImageNetSifter
import uuid

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1153006
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 128161
num_preprocess_threads = 24
training_batch_size = 200
generation_batch_size = training_batch_size
training_epochs = 1
img_size = 96
num_params = 512
inc_name = 'Xception'
first_layer_dim = 10
color_channels = 3
epochs_gen_ratio = 1
epochs = 1000
num_samples = 200
imagenet_dir = './ImageNetImages'
model_folder = './Models'
model_name = inc_name + 'Checkpoint'
pre_name = 'Pre' + model_name
# num_pics = len(os.listdir(directory + "\\Snakes"))

@tf.autograph.experimental.do_not_convert
def initializeSnakeIdentifier(
        train_datagen,
        validation_datagen
    ):
    print('Starting from Scratch')
    
    train_df = pd.read_csv('./classes_train.csv')
    validate_df = pd.read_csv('./classes_validate.csv')

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file',
        y_col='snake',
        target_size=(img_size, img_size),
        batch_size=training_batch_size,
        class_mode='binary',
        validate_filenames=False
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validate_df,
        x_col='file',
        y_col='snake',
        target_size=(img_size, img_size),
        batch_size=training_batch_size,
        class_mode='binary',
        validate_filenames=False
    )
    # build the Xception network
    model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    # model.save_weights(model_folder + '/' + pre_name + "PristineNoWeight.hdf5")

    # set everything except the last separable convolution block to be untrainable
    for layer in model.layers[:-4]:
        layer.trainable = False
    #model = load_model(pre_name + ".hdf5")
    print('Model loaded.')
    # build a classifier model to put on top of the convolutional model
    top_model = GlobalAveragePooling2D()(model.output)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(1, activation='sigmoid')(top_model)

    #model =  Model(inputs=[main_input], outputs=[top_model])

    # add the model on top of the convolutional base
    model = Model(inputs = model.input, outputs=top_model)
    # for layer in model.layers:
    #     print(layer.dtype)
    # model.summary()

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(epsilon=K.epsilon()),
        metrics=['accuracy']
    )

    # fine-tune the model
    model.fit(
        train_generator,
        steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // training_batch_size,
        epochs=training_epochs,
        validation_data=validation_generator,
        validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // training_batch_size,
        callbacks=[ModelCheckpoint(model_folder + '/' + model_name + '.hdf5', save_best_only=True, mode = 'min')]
    )
    print("Trained Top")
    # model.predict(validation_generator)

@tf.autograph.experimental.do_not_convert
def createSnekMaker():
    main_input = Input(shape=(num_params,), dtype='float32', name='main_input')
    x = Dense(num_params * first_layer_dim * first_layer_dim, activation='relu')(main_input)
    #x = Dropout(.4)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Reshape((first_layer_dim, first_layer_dim, num_params))(x)
    x = Conv2DTranspose(256, kernel_size = 4, strides = 2)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU()(x)
    x = Conv2DTranspose(128, kernel_size = 4, strides = 2)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU()(x)
    x = Conv2DTranspose(64, kernel_size = 4, strides = 2)(x)
    # x = Conv2DTranspose(64, kernel_size = 4, strides = 1)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU()(x)
    x = Conv2DTranspose(32, kernel_size = 3, strides = 1)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU()(x)
    x = Conv2DTranspose(16, kernel_size = 3, strides = 1)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU()(x)
    New_img = Conv2D(color_channels, kernel_size = 3, activation=tanh)(x)
    gen_model = Model(main_input, New_img)
    gen_model.summary()
    return gen_model

@tf.autograph.experimental.do_not_convert
def trainSnekMaker(
        train_datagen,
        num_training_samples=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
        num_validation_samples=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL,
        gen_model=None,
        check_model=None,
        train_model=None
    ):
    if not train_model:
        train_model = Sequential()
        train_model.add(gen_model)
        train_model.add(check_model)
    if check_model:
        check_model.summary()
    if gen_model:
        gen_model.summary()
    train_model.summary()
    train_model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(epsilon=K.epsilon()),
        metrics=['accuracy']
    )
    for x in range(0, epochs):
        gan_uuid = uuid.uuid4()
        train_model.layers[0].trainable = True
        train_model.layers[1].trainable = False

        input_values = np.random.random_sample((num_samples, num_params))
        train_model.fit(
            input_values,
            np.ones(num_samples),
            batch_size=generation_batch_size,
            callbacks=[ModelCheckpoint('{}/SnekTrainerWhole.hdf5'.format(model_folder))],
            epochs=(x+1),
            initial_epoch=x
        )
        
        train_model.layers[0].save_weights(
            '{}/SnekGAN_{}.hdf5'.format(model_folder, gan_uuid),
            overwrite=True
        )

        snek_gen_model = train_model.layers[0]
        fake_snakes = input_values[:1000]
        output_squares = snek_gen_model.predict(
            input_values,
            batch_size=generation_batch_size,
            callbacks=[ProgbarLogger()]
        )
        output_folder = '{}/fake_snakes_{}'.format(ImageNetSifter.imagenet_dir, gan_uuid)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for y in range(0, len(output_squares)):
            array_to_img(output_squares[y]).save('{}/snake_{}.jpeg'.format(output_folder, y))
        
        test_values = input_values[-5:]
        print(train_model.predict(test_values))
        num_training_samples, num_validation_samples = ImageNetSifter.decodeDir(only_snakes=True)

        train_df = pd.read_csv('./classes_train.csv')
        validate_df = pd.read_csv('./classes_validate.csv')
        
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='file',
            y_col='snake',
            target_size=(img_size, img_size),
            batch_size=generation_batch_size,
            class_mode='binary',
            validate_filenames=False
        )
        
        validation_generator = train_datagen.flow_from_dataframe(
            dataframe=validate_df,
            x_col='file',
            y_col='snake',
            target_size=(img_size, img_size),
            batch_size=generation_batch_size,
            class_mode='binary',
            validate_filenames=False
        )
        # print(len(validation_generator))
        
        train_model.layers[0].trainable = False
        train_model.layers[1].trainable = True
        # num_layers_id = len(check_model.layers)
        # train_model.summary()
        train_model.layers[-1].fit(
            validation_generator,
            # steps_per_epoch=1,
            steps_per_epoch=num_validation_samples // generation_batch_size,
            epochs=int(x+1),
            callbacks=[ModelCheckpoint('{}/CheckerTry_{}.hdf5'.format(model_folder, gan_uuid), save_best_only=False, mode = 'min')],
            initial_epoch = int(x)
        )
        print(train_model.predict(test_values))

@tf.autograph.experimental.do_not_convert
def main(
        start_from_scratch=False,
        fine_tune=False,
        use_mixed_precision=False,
        gan_id=False,
        training_batch_size=training_batch_size,
        generation_batch_size=generation_batch_size
    ):
    # print(tf.__version__)
    # print(tf.config.list_physical_devices('GPU'))
    if use_mixed_precision:
        print('Using Mixed Precision')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    else:
        policy = mixed_precision.Policy('float32')
        mixed_precision.set_policy(policy)
    epsilon = 1e-7
    dtype = 'float32'
    K.set_epsilon(epsilon)
    K.set_floatx(dtype)
    print(K.floatx(), K.epsilon(), training_batch_size)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    tf.autograph.set_verbosity(0, False)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    if start_from_scratch:
        initializeSnakeIdentifier(
            train_datagen,
            test_datagen
        )
    snek_generator = createSnekMaker()
    
    train_model = None
    if gan_id:
        print('Loading ', gan_id)
        train_model = load_model('{}/CheckerTry_{}.hdf5'.format(model_folder, gan_id))

    snek_checker = load_model('Models/' + model_name+".hdf5")
    snek_checker.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )
    trainSnekMaker(
        train_datagen,
        check_model=snek_checker,
        gen_model=snek_generator,
        train_model=train_model
    )
    
    # if fine_tune or start_from_scratch:
    #     fineTuneSnekMaker(
    #         train_datagen,
    #         test_datagen,
    #         train_generator,
    #         validation_generator
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Snake Making GAN.')
    parser.add_argument(
        '--fine-tune', 
        help='Fine tune the pre trained model',
        action='store_true'
    )
    parser.add_argument(
        '--start-from-scratch',
        help='Retrain the whole XCeptionModel from the start',
        action='store_true'
    )
    parser.add_argument(
        '--use-mixed-precision',
        help='Used mixed precision',
        action='store_true'
    )
    parser.add_argument(
        '-g',
        '--gan-id',
        help='GAN Id to load',
    )
    args = parser.parse_args()
    main(
        start_from_scratch=args.start_from_scratch,
        fine_tune=args.fine_tune,
        use_mixed_precision=args.use_mixed_precision,
        gan_id=args.gan_id
    )