# -*- coding: utf-8 -*-
"""
Created on Sat May 16 00:25:28 2020

@author: smoss

This is a gan inspired by Big GAN and based on the implementation here https://github.com/taki0112/BigGAN-Tensorflow
"""
import math
import time

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
import imagenetLabels

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
TOTAL_EXAMPLES = 1281166
TRAINING_BATCH_SIZE = 16
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = TOTAL_EXAMPLES #- (TOTAL_EXAMPLES % TRAINING_BATCH_SIZE)
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 0
num_preprocess_threads = 24
generation_batch_size = TRAINING_BATCH_SIZE
training_epochs = 1
# IMG_SIZE = 128
NUM_PARAMS = 128
inc_name = 'Xception'
first_layer_dim = 10
color_channels = 3
epochs_gen_ratio = 1
EPOCHS = 1000
num_samples = 200
imagenet_dir = './ImageNetImages'
model_folder = './Models'
model_name = inc_name + 'Checkpoint'
pre_name = 'Pre' + model_name
# num_pics = len(os.listdir(directory + "\\Snakes"))
generator_optimizer = tf.keras.optimizers.SGD(momentum=.5, nesterov=True)
discriminator_optimizer = tf.keras.optimizers.SGD(.0002, momentum=.5, nesterov=True)
CHANNEL_MULT = 80
CHANNELS = 16 * CHANNEL_MULT
CLASS_PARAMS = CHANNEL_MULT
INIT_SIZE = (4, 4, 1)
OUTPUT_CLASSES = 1000
FINAL_IMG_SIZE = 64
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.autograph.experimental.do_not_convert
def initializeSnakeIdentifier(num_params=NUM_PARAMS, channels=CHANNELS):
    curr_channels = CHANNEL_MULT
    class_in = layers.Input(shape=(1,))
    embedding_layer = layers.Embedding(OUTPUT_CLASSES, CHANNELS)(class_in)

    img_in = layers.Input(shape=(FINAL_IMG_SIZE, FINAL_IMG_SIZE, 3))
    disc = CustomLayers.Conv2D(curr_channels, kernel_size=3, padding='same')(img_in)

    # curr_channels *= 2
    # disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlockDown1', down=True)(disc)
    # disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlock1')(disc)
    disc = CustomLayers.SoftAttentionMax(curr_channels)(disc)

    curr_channels *= 4
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlockDown2', down=True)(disc)
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlock2')(disc)

    curr_channels *= 2
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlockDown3', down=True)(disc)
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlock3')(disc)

    curr_channels *= 2
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlockDown4', down=True)(disc)
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlock4')(disc)

    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlockDown5', down=True)(disc)
    disc = CustomLayers.ResBlockCondDownD(curr_channels, name='DiscBlock5')(disc)

    disc = layers.ReLU()(disc)
    disc = CustomLayers.GlobalSumPooling2D()(disc)

    embedding_layer = layers.Flatten()(embedding_layer)
    embedding_layer = layers.Dot((1, 1))([disc, embedding_layer])

    disc = CustomLayers.Dense(1)(disc)

    judge = layers.Add(name='FinalAdd')(
        [
            disc,
            embedding_layer
        ]
    )

    discriminator = Model(inputs=[img_in, class_in], outputs=judge, name='Discriminator')

    discriminator.summary()
    return discriminator

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
def createSnekMaker(
        num_params=NUM_PARAMS,
        channels=CHANNELS,
        init_size=INIT_SIZE
):

    curr_channels = channels
    noise_in = layers.Input((num_params,))
    embedding_in = layers.Input(shape=(1,))
    embedding_layer = layers.Embedding(OUTPUT_CLASSES, CLASS_PARAMS)(embedding_in)
    embedding_layer = layers.Flatten()(embedding_layer)

    repeat_layer = layers.Concatenate()([noise_in, embedding_layer])

    gen = CustomLayers.Dense(init_size[0] * init_size[1] * curr_channels, use_bias=False)(repeat_layer)
    gen = layers.BatchNormalization()(gen)
    gen = layers.ReLU()(gen)
    gen = layers.Reshape((init_size[0], init_size[1], curr_channels))(gen)

    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock1', split=False)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp1', split=False, up=True)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock2', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp2', up=True)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock3', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp3', up=True)([gen, repeat_layer])
    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock4', split=False)([gen, repeat_layer])
    curr_channels = curr_channels // 2

    gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp4', up=True)([gen, repeat_layer])
    gen = CustomLayers.SoftAttentionMax(curr_channels)(gen)
    # gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlock5', split=False)([gen, repeat_layer])
    # curr_channels = curr_channels // 2

    # gen = CustomLayers.ResBlockCondD(curr_channels, name='ResBlockUp5', up=True)([gen, repeat_layer])

    gen = layers.BatchNormalization(momentum=.9)(gen)

    gen = layers.ReLU()(gen)

    gen = layers.ZeroPadding2D()(gen)
    new_img = CustomLayers.Conv2D(color_channels, kernel_size=3, activation=tanh)(gen)
    gen_model = Model([noise_in, embedding_in], new_img, name='Generator')
    gen_model.summary()
    # noise, cats = generate_fake_images()
    # import time
    # start = time.time()
    # baby_noise = gen_model.predict([noise, cats], batch_size=generation_batch_size)
    # print('Time to predict is {} sec'.format(time.time() - start))
    # noise, cats = generate_fake_images()
    # start = time.time()
    # baby_noise = gen_model.predict([noise, cats], batch_size=generation_batch_size)
    # print('Time to predict is {} sec'.format(time.time() - start))
    # saveFakes(baby_noise)
    return gen_model#, baby_noise

def generate_and_save_images(model, epoch, num_params=NUM_PARAMS, num_images=2):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm)
    start = time.time()
    for idx in range(1000):
        noise = tf.random.normal([num_images, num_params])
        categories = tf.constant([idx for _ in range(num_images)])
        predictions = model([noise, categories], training=False)
        output_folder = 'big_gan_fakes/fake_imgs_{}'.format(epoch)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for y in range(0, len(predictions)):
            tf.keras.preprocessing.image.array_to_img(predictions[y]).\
                save('{}/{}_id_{}.jpeg'.format(
                    output_folder,
                    imagenetLabels.imagenet_labels[idx+1][0],
                    y
                ))

    print('Took ', time.time() - start, ' seconds')

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, labels, generator, discriminator):
    labels = tf.reshape(labels, (-1,))
    images = tf.reshape(images, (-1, FINAL_IMG_SIZE, FINAL_IMG_SIZE, 3))
    generated_noise_1, generated_labels_1 = generate_fake_images(images.shape[0])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator([generated_noise_1, generated_labels_1], training=True)

      real_output = discriminator([images, labels], training=True)
      fake_output = discriminator([generated_images, generated_labels_1], training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(train_df, epochs, generator, discriminator, gan, checkpoint, checkpoint_prefix):
    for epoch in range(epochs):
        epoch += 1

        tot_gen_loss = 0
        tot_disc_loss = 0
        data_len = math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / TRAINING_BATCH_SIZE)
        dataset = make_dataset(train_df)
        for image_batch, image_labels in tqdm(
                dataset,
                desc='Epoch {}'.format(epoch),
                total=data_len
        ):
            # This is significantly faster than using train on batch because we don't have to
            # double run discriminator
            train_step(image_batch, image_labels, generator, discriminator)
            # with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            #
            #     real_output = discriminator(
            #         [image_batch, image_labels],
            #         training=True
            #     )
            #
            #     generated_noise_1, generated_labels_1 = generate_fake_images(image_batch.shape[0])
            #     generated_images = generator([generated_noise_1, generated_labels_1], training=True)
            #     fake_output = discriminator([generated_images, generated_labels_1], training=True)
            #
            #     gen_loss = generator_loss(fake_output)
            #     disc_loss = discriminator_loss(real_output, fake_output)
            #
            # gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            # gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            #
            # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            # discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            # Save the model every 15 epochs
        # if epoch % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

        generator.save('Models/BIGGAN_Sneks.hdf5')

        # if epoch % 2 == 1:
        generate_and_save_images(generator,
                                 epoch)
        print(tot_gen_loss / data_len, tot_disc_loss / data_len)

    # Generate after the final epoch
    generate_and_save_images(generator,
                             'final')

def make_gan(discriminator, generator):
    discriminator.trainable = False
    gen_noise, gen_label = generator.input

    gen_image = generator.output

    gan_output = discriminator([gen_image, gen_label])

    model = tf.keras.Model([gen_noise, gen_label], gan_output)

    model.compile(loss="binary_crossentropy", optimizer=generator_optimizer)
    return model

def make_dataset(train_df):
    # train_csv_rows, _ = ImageNetSifter.decodeDir()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    # start = time.time()
    # train_df = train_df.sample(frac=1).reset_index(drop=True)
    # print('Took ', time.time() - start, ' seconds')
    train_generator = lambda : train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file',
        y_col='class_num',
        target_size=(FINAL_IMG_SIZE, FINAL_IMG_SIZE),
        validate_filenames=False,
        batch_size=1,
        class_mode='sparse'
    )
    # imgs, classes = next(validation_generator)
    # print(imgs.shape)
    # print(classes.shape)
    return tf.data.Dataset.from_generator(
        train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([1, FINAL_IMG_SIZE, FINAL_IMG_SIZE, 3], [1])
    ).take(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN).batch(TRAINING_BATCH_SIZE)

@tf.autograph.experimental.do_not_convert
def main(
        use_mixed_precision=False,
        training_batch_size=TRAINING_BATCH_SIZE,
        generation_batch_size=generation_batch_size,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    ):
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

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # if start_from_scratch:
    #     initializeSnakeIdentifier(
    #         train_datagen,
    #         test_datagen
    #     )
    snek_generator = createSnekMaker()
    snek_discriminator = initializeSnakeIdentifier()
    # snek_discriminator.predict([baby_noise, tf.constant([0]*32)])
    gan = make_gan(snek_discriminator, snek_generator)
    snek_discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

    train_df = pd.read_csv('./classes_train.csv')

    checkpoint_dir = './snek_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    print(checkpoint_prefix)
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        snek_checker=snek_discriminator,
        snek_generator=snek_generator,
        gan=gan
    )
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # checkpoint.save(file_prefix=checkpoint_prefix)
    # trainSnekMaker(
    #     train_datagen,
    #     check_model=snek_checker,
    #     gen_model=snek_generator,
    #     train_model=train_model
    # )
    train(
        train_df,
        EPOCHS,
        snek_generator,
        snek_discriminator,
        gan,
        checkpoint,
        checkpoint_prefix
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Snake Making GAN.')
    parser.add_argument(
        '--use-mixed-precision',
        help='Used mixed precision',
        action='store_true'
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
        generator_optimizer = tf.keras.optimizers.Adam(5e-5, beta_1=0)
        discriminator_optimizer = tf.keras.optimizers.Adam(3e-4, beta_1=0)
    elif args.optimizer == 'nadam':
        print('Using Nadam optimizer')
        generator_optimizer = tf.keras.optimizers.Nadam(5e-5, beta_1=0)
        discriminator_optimizer = tf.keras.optimizers.Nadam(2e-4, beta_1=0)
    else:
        print('Using SGD optimizer')
    main(
        use_mixed_precision=args.use_mixed_precision,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )