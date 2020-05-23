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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
from tensorflow.keras import optimizers
from pprint import pprint
import SplitImages

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 18656 * 3
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2073 * 3
num_preprocess_threads = 16
batch_size = 400
epochs = 20
img_size = 150
inc_name = 'Xception'
model_name = inc_name + 'Wunderkid2'
pre_name = 'Pre' + model_name
# num_pics = len(os.listdir(directory + "\\Snakes"))

@tf.autograph.experimental.do_not_convert
def main():
    # print(tf.__version__)
    # print(tf.config.list_physical_devices('GPU'))
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)
    tf.autograph.set_verbosity(0, False)

    # build the VGG16 network
    model = Xception(weights='imagenet', include_top=False, 
        input_shape = (img_size, img_size, 3))
    model.save(pre_name + "PristineNoWeight.hdf5")
    #model = load_model(pre_name + ".hdf5")
    print('Model loaded.')
    # build a classifier model to put on top of the convolutional model
    top_model = GlobalAveragePooling2D()(model.output)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dropout(0.5)(top_model)
    top_model = Dense(1, activation='sigmoid')(top_model)

    #model =  Model(inputs=[main_input], outputs=[top_model])
    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:-4]:
        layer.trainable = False

    # add the model on top of the convolutional base
    model = Model(inputs = model.input, outputs=top_model)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'FirstRoundTraining',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        'SecondRoundTraining',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model.fit(
        train_generator,
        steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size,
        epochs=epochs//5,
        validation_data=validation_generator,
        validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // batch_size,
        callbacks=[ModelCheckpoint(pre_name + '.hdf5', save_best_only=True, mode = 'min')])
    print("Trained Top")
    
    model = load_model(pre_name + '.hdf5')
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',
                   metrics=['accuracy'])
    # fine-tune the model
    model.fit(
        train_generator,
        steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // batch_size,
        callbacks=[ModelCheckpoint(model_name + '.hdf5', save_best_only=True, mode = 'min')])

if __name__ == "__main__":
    main()