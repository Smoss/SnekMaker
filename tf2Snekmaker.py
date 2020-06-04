# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:25:28 2020

@author: smoss

This is a gan inspired by Big GAN and based on the implementation here https://github.com/taki0112/BigGAN-Tensorflow
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
import CustomLayers
import random

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1153006
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 128161
num_preprocess_threads = 24
training_batch_size = 200
generation_batch_size = training_batch_size
training_epochs = 1
IMG_SIZE = 128
NUM_PARAMS = 128
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
generator_optimizer = tf.keras.optimizers.SGD(momentum=.5, nesterov=True)
discriminator_optimizer = tf.keras.optimizers.SGD(.0002,momentum=.5, nesterov=True)
CHANNEL_MULT = 128
CHANNELS = 16 * CHANNEL_MULT
CLASS_PARAMS = CHANNEL_MULT
INIT_SIZE = (4, 4, 1)

@tf.autograph.experimental.do_not_convert
def initializeSnakeIdentifier(
        train_datagen,
        validation_datagen
    ):
    pass
    # print('Starting from Scratch')

    # print("Trained Top")

def saveFakes(images, parent='Fakes', folder='tryout'):
        output_folder = '{}/{}'.format(parent, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for y in range(0, len(images)):
            array_to_img(images[y]).save('{}/fake_{}.jpeg'.format(output_folder, y))

def generate_fake_images(count=generation_batch_size, noise_dim=NUM_PARAMS, n_classes=1000):
    noise = tf.random.normal([count, noise_dim])
    categories = tf.constant([random.randint(0, n_classes - 1) for _ in range(count)])
    return noise, categories

@tf.autograph.experimental.do_not_convert
def createSnekMaker(num_params=NUM_PARAMS, channels=CHANNELS, init_size=INIT_SIZE):
    # base_size = num_params // 6
    # split_rem = num_params % base_size
    # print(num_params % base_size, num_params, base_size)
    # if split_rem == 0:
    #     noise_split_list = [base_size] * 6
    # else:
    #     noise_split_list = [base_size] * 5 + [split_rem + base_size]

    curr_channels = channels
    noise_in = layers.Input((num_params,))
    embedding_in = layers.Input(shape=(1,))
    embedding_layer = layers.Embedding(1000, CLASS_PARAMS)(embedding_in)
    embedding_layer = layers.Flatten()(embedding_layer)

    repeat_layer = layers.Concatenate()([noise_in, embedding_layer])

    gen = layers.Dense(init_size[0] * init_size[1] * curr_channels, use_bias=False)(repeat_layer)
    gen = layers.BatchNormalization()(gen)
    gen = layers.LeakyReLU()(gen)
    gen = layers.Reshape((init_size[0], init_size[1], curr_channels))(gen)

    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock1', split=False)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp1', split=False, up=True)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock2', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    print(gen)
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp2', up=True)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock3', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    print(gen)
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp3', up=True)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock4', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    print(gen)
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp4', up=True)([gen, repeat_layer])
    gen = CustomLayers.SoftAttentionMax(curr_channels)(gen)
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock5', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    print(gen)
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp5', up=True)([gen, repeat_layer])
    print(gen)

    gen = layers.BatchNormalization(momentum=.9)(gen)

    gen = layers.LeakyReLU()(gen)

    gen = layers.ZeroPadding2D()(gen)
    new_img = CustomLayers.Conv2D(color_channels, kernel_size=3, activation=tanh)(gen)
    gen_model = Model([noise_in, embedding_in], new_img)
    gen_model.summary()
    noise, cats = generate_fake_images()
    import time
    K.clear_session()
    start = time.time()
    baby_noise = gen_model.predict([noise, cats], batch_size=generation_batch_size)
    print('Time to predict is {} sec'.format(time.time() - start))
    noise, cats = generate_fake_images()
    start = time.time()
    baby_noise = gen_model.predict([noise, cats], batch_size=generation_batch_size)
    print('Time to predict is {} sec'.format(time.time() - start))
    print(baby_noise.shape)
    saveFakes(baby_noise)
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
    pass
    # if not train_model:
    #     train_model = Sequential()
    #     train_model.add(gen_model)
    #     train_model.add(check_model)
    # if check_model:
    #     check_model.summary()
    # if gen_model:
    #     gen_model.summary()
    # train_model.summary()
    # train_model.compile(
    #     loss='binary_crossentropy',
    #     optimizer=optimizers.Adam(epsilon=K.epsilon()),
    #     metrics=['accuracy']
    # )
    # for x in range(0, epochs):
    #     gan_uuid = uuid.uuid4()
    #     train_model.layers[0].trainable = True
    #     train_model.layers[1].trainable = False
    #
    #     input_values = np.random.random_sample((num_samples, num_params))
    #     train_model.fit(
    #         input_values,
    #         np.ones(num_samples),
    #         batch_size=generation_batch_size,
    #         callbacks=[ModelCheckpoint('{}/SnekTrainerWhole.hdf5'.format(model_folder))],
    #         epochs=(x+1),
    #         initial_epoch=x
    #     )
    #
    #     train_model.layers[0].save_weights(
    #         '{}/SnekGAN_{}.hdf5'.format(model_folder, gan_uuid),
    #         overwrite=True
    #     )
    #
    #     snek_gen_model = train_model.layers[0]
    #     fake_snakes = input_values[:1000]
    #     output_squares = snek_gen_model.predict(
    #         input_values,
    #         batch_size=generation_batch_size,
    #         callbacks=[ProgbarLogger()]
    #     )
    #     output_folder = '{}/fake_snakes_{}'.format(ImageNetSifter.imagenet_dir, gan_uuid)
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #
    #     for y in range(0, len(output_squares)):
    #         array_to_img(output_squares[y]).save('{}/snake_{}.jpeg'.format(output_folder, y))
    #
    #     test_values = input_values[-5:]
    #     print(train_model.predict(test_values))
    #     num_training_samples, num_validation_samples = ImageNetSifter.decodeDir(only_snakes=True)
    #
    #     train_df = pd.read_csv('./classes_train.csv')
    #     validate_df = pd.read_csv('./classes_validate.csv')
    #
    #     train_generator = train_datagen.flow_from_dataframe(
    #         dataframe=train_df,
    #         x_col='file',
    #         y_col='snake',
    #         target_size=(img_size, img_size),
    #         batch_size=generation_batch_size,
    #         class_mode='binary',
    #         validate_filenames=False
    #     )
    #
    #     validation_generator = train_datagen.flow_from_dataframe(
    #         dataframe=validate_df,
    #         x_col='file',
    #         y_col='snake',
    #         target_size=(img_size, img_size),
    #         batch_size=generation_batch_size,
    #         class_mode='binary',
    #         validate_filenames=False
    #     )
    #     # print(len(validation_generator))
    #
    #     train_model.layers[0].trainable = False
    #     train_model.layers[1].trainable = True
    #     # num_layers_id = len(check_model.layers)
    #     # train_model.summary()
    #     train_model.layers[-1].fit(
    #         validation_generator,
    #         # steps_per_epoch=1,
    #         steps_per_epoch=num_validation_samples // generation_batch_size,
    #         epochs=int(x+1),
    #         callbacks=[ModelCheckpoint('{}/CheckerTry_{}.hdf5'.format(model_folder, gan_uuid), save_best_only=False, mode = 'min')],
    #         initial_epoch = int(x)
    #     )
    #     print(train_model.predict(test_values))

@tf.autograph.experimental.do_not_convert
def main(
        start_from_scratch=False,
        fine_tune=False,
        use_mixed_precision=False,
        gan_id=False,
        training_batch_size=training_batch_size,
        generation_batch_size=generation_batch_size,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
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

    # if start_from_scratch:
    #     initializeSnakeIdentifier(
    #         train_datagen,
    #         test_datagen
    #     )
    snek_generator = createSnekMaker()
    
    train_model = None
    # if gan_id:
    #     print('Loading ', gan_id)
    #     train_model = load_model('{}/CheckerTry_{}.hdf5'.format(model_folder, gan_id))

    # snek_checker = load_model('Models/' + model_name+".hdf5")
    # snek_checker.compile(
    #     loss='binary_crossentropy',
    #     optimizer='Adam',
    #     metrics=['accuracy']
    # )
    # checkpoint_dir = './snek_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
    #                                  discriminator_optimizer=discriminator_optimizer,
    #                                  snek_checker=snek_checker,
    #                                  snek_generator=snek_generator)
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # trainSnekMaker(
    #     train_datagen,
    #     check_model=snek_checker,
    #     gen_model=snek_generator,
    #     train_model=train_model
    # )
    
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
    parser.add_argument(
        '-o',
        '--optimizer',
        help='optimizier to user',
        default='SGD'
    )
    args = parser.parse_args()
    if args.optimizer == 'adam':
        print('Using Adam optimizer')
        generator_optimizer = tf.keras.optimizers.Adam(beta_1=.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
    elif args.optimizer == 'nadam':
        print('Using Nadam optimizer')
        generator_optimizer = tf.keras.optimizers.Nadam(beta_1=.5)
        discriminator_optimizer = tf.keras.optimizers.Nadam(2e-4, beta_1=.5)
    else:
        print('Using SGD optimizer')
    main(
        start_from_scratch=args.start_from_scratch,
        fine_tune=args.fine_tune,
        use_mixed_precision=args.use_mixed_precision,
        gan_id=args.gan_id,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )