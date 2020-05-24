# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:25:28 2020

@author: smoss
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential, load_model, Model
import cv2
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
from tensorflow.keras import optimizers
from pprint import pprint
import SplitImages
import argparse

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 72656 * 3
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 8073 * 3
num_preprocess_threads = 16
training_batch_size = 100
generation_batch_size = 100
training_epochs = 1
img_size = 200
inc_name = 'Xception'
model_name = inc_name + 'Wunderkid2'
pre_name = 'Pre' + model_name
# num_pics = len(os.listdir(directory + "\\Snakes"))

@tf.autograph.experimental.do_not_convert
def initializeSnakeIdentifier(
        train_datagen,
        test_datagen,
        train_generator,
        validation_generator
    ):
    print('Starting from Scratch')
    # build the Xception network
    model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    model.save(pre_name + "PristineNoWeight.hdf5")

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
        callbacks=[ModelCheckpoint(pre_name + '.hdf5', save_best_only=True, mode = 'min')])
    print("Trained Top")

@tf.autograph.experimental.do_not_convert
def createSnekMaker():
    pass
# @tf.autograph.experimental.do_not_convert
# def fineTuneSnekMaker(
#         train_datagen,
#         test_datagen,
#         train_generator,
#         validation_generator
#     ):
#     print('Fine tuning the model')
#     model = load_model(pre_name + '.hdf5')
#     for layer in model.layers[:172]:
#         layer.trainable = False
#     for layer in model.layers[172:]:
#         layer.trainable = True

#     model.compile(
#         optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     # fine-tune the model
#     model.fit(
#         train_generator,
#         steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // training_batch_size,
#         epochs=training_epochs,
#         validation_data=validation_generator,
#         validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // training_batch_size,
#         callbacks=[ModelCheckpoint(model_name + '.hdf5', save_best_only=True, mode = 'min')])
    

@tf.autograph.experimental.do_not_convert
def main(
        start_from_scratch=False,
        fine_tune=False,
        use_mixed_precision=False,
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
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'FirstRoundTraining',
        target_size=(img_size, img_size),
        batch_size=training_batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        'SecondRoundTraining',
        target_size=(img_size, img_size),
        batch_size=training_batch_size,
        class_mode='binary'
    )

    if start_from_scratch:
        initializeSnakeIdentifier(
            train_datagen,
            test_datagen,
            train_generator,
            validation_generator
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
    args = parser.parse_args()
    main(
        start_from_scratch=args.start_from_scratch,
        fine_tune=args.fine_tune,
        use_mixed_precision=args.use_mixed_precision
    )