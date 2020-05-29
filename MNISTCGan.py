import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf
import os

tf.autograph.set_verbosity(0, False)
from IPython import display
from tqdm import tqdm
import math
import random

BUFFER_SIZE = 60000
BATCH_SIZE = 1000
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPOCHS = 500
noise_dim = 100
num_examples_to_generate = 16
# generator_optimizer = tf.keras.optimizers.Adam(beta_1=.5)
# discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=.5)
generator_optimizer = tf.keras.optimizers.SGD(.1,momentum=.5, nesterov=True)
discriminator_optimizer = tf.keras.optimizers.SGD(.02, momentum=.5, nesterov=True)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def make_generator_model(input_dim=noise_dim, n_classes=10):
    label_in = layers.Input(shape=(1,))

    li = layers.Embedding(n_classes, noise_dim)(label_in)

    li = layers.Dense(7 * 7)(li)
    li = layers.Reshape((7, 7, 1))(li)
    # print(li.shape)
    # assert li.shape == (None, 7, 7, 1)

    noise_in = layers.Input((input_dim,))

    gen = layers.Dense(7 * 7 * 256, use_bias=False)(noise_in)
    gen = layers.BatchNormalization()(gen)
    gen = layers.LeakyReLU()(gen)

    gen = layers.Reshape((7, 7, 256))(gen)
    # assert gen.shape == (None, 7, 7, 256) # Note: None is the batch size

    merge = layers.Concatenate()([gen, li])
    gen = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    # assert gen.shape == (None, 7, 7, 128)
    gen = layers.BatchNormalization()(gen)
    gen = layers.LeakyReLU()(gen)

    gen = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(gen)
    # assert gen.shape == (None, 14, 14, 64)
    gen = layers.BatchNormalization()(gen)
    gen = layers.LeakyReLU()(gen)

    fake_image = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(
        gen)
    # assert fake_image.shape == (None, 28, 28, 1)

    model = tf.keras.Model([noise_in, label_in], fake_image)
    # print(model.output_shape)
    # model.summary()
    return model


def make_discriminator_model(in_shape=(28, 28, 1), n_classes=10):
    label_in = layers.Input(shape=(1,))
    # Create the category vector
    ll = layers.Embedding(n_classes, noise_dim)(label_in)
    # Create a tensor we can concatenate
    ll = layers.Dense(in_shape[0] * in_shape[1])(ll)
    ll = layers.Reshape(in_shape)(ll)
    image_in = layers.Input(shape=in_shape)
    merge = layers.Concatenate()([image_in, ll])
    # Downsample
    dl = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merge)
    dl = layers.LeakyReLU()(dl)
    dl = layers.Dropout(0.3)(dl)
    # Downsample
    dl = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(dl)
    dl = layers.LeakyReLU()(dl)
    dl = layers.Dropout(0.3)(dl)
    # Discriminate
    dl = layers.Flatten()(dl)
    disc = layers.Dense(1)(dl)

    model = tf.keras.Model([image_in, label_in], disc)
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
    return model


def make_gan(discriminator, generator):
    discriminator.trainable = False
    gen_noise, gen_label = generator.input

    gen_image = generator.output

    gan_output = discriminator([gen_image, gen_label])

    model = tf.keras.Model([gen_noise, gen_label], gan_output)

    model.compile(loss="binary_crossentropy", optimizer=generator_optimizer)
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(real_output), fake_output)

    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, num_images=50):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm)
    noise = tf.random.normal([num_images, noise_dim])
    categories = tf.constant([x // (num_images // 10) for x in range(num_images)])
    predictions = model([noise, categories], training=False)
    output_folder = 'mnist_fakes/fake_nums_{}'.format(epoch)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for y in range(0, len(predictions)):
        tf.keras.preprocessing.image.array_to_img(predictions[y]).save('{}/{}_id_{}.jpeg'.format(output_folder, categories[y], y))

    # fig = plt.figure(figsize=(4,4))
    #
    # for i in range(predictions.shape[0]):
    #   plt.subplot(4, 4, i+1)
    #   plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    #   plt.axis('off')
    #
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def generate_fake_images(count=BATCH_SIZE):
    noise = tf.random.normal([count, noise_dim])
    categories = tf.constant([random.randint(0, 9) for _ in range(count)])
    return noise, categories


def train(dataset, epochs, generator, discriminator, gan, checkpoint, checkpoint_prefix):
    for epoch in range(epochs):
        epoch += 1

        tot_gen_loss = 0
        tot_disc_loss = 0
        data_len = math.ceil(BUFFER_SIZE / BATCH_SIZE)
        for image_batch, image_labels in tqdm(
                dataset,
                desc='Epoch {}'.format(epoch),
                total=data_len
        ):
            disc_loss_1, _ = discriminator.train_on_batch(
                [image_batch, image_labels],
                tf.ones_like(image_labels)
            )
            generated_noise_1, generated_labels_1 = generate_fake_images()
            generated_images = generator([generated_noise_1, generated_labels_1], training=True)
            disc_loss_2, _ = discriminator.train_on_batch(
                [generated_images, generated_labels_1],
                tf.zeros_like(generated_labels_1)
            )
            # generated_noise_2, generated_labels_2 = generate_fake_images()
            # print(generated_noise, generated_labels)
            gen_loss = gan.train_on_batch(
                [generated_noise_1, generated_labels_1],
                tf.ones_like(generated_labels_1)
            )
            tot_gen_loss += gen_loss
            tot_disc_loss += disc_loss_1 + disc_loss_2
            # Save the model every 15 epochs
        print(tot_gen_loss / data_len, tot_disc_loss / data_len)
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        generator.save('Models/Mnist_CGAN.hdf5')

        # Generate after the final epoch
        generate_and_save_images(generator,
                                 epoch)


def main():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # print(len(train_dataset))

    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    train_dataset = zip(train_images, train_labels)
    # print(train_images.shape, train_labels.shape)
    # print(tf.data.Dataset.from_tensor_slices([tf.ones_like((2, 2, 2)), tf.zeros_like((2, 2))]))
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE)
    generator = make_generator_model()

    discriminator = make_discriminator_model()
    gan = make_gan(discriminator, generator)
    # discriminator.summary()
    # gan.summary()
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    train(train_dataset, EPOCHS, generator, discriminator, gan, checkpoint, checkpoint_prefix)


if __name__ == "__main__":
    main()



