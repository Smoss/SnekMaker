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
FINAL_IMG_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1281167
def main(final_img_size=FINAL_IMG_SIZE):
    train_df = pd.read_csv('./classes_train.csv')
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )
    train_generator = lambda: train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='file',
        y_col='class_num',
        target_size=(final_img_size, final_img_size),
        validate_filenames=False,
        batch_size=128
    )
    # imgs, classes = next(validation_generator)
    # print(imgs.shape)
    # print(classes.shape)
    dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([32, FINAL_IMG_SIZE, FINAL_IMG_SIZE, 3], [32])
    )

    img_array = np.ndarray(shape=(0, 128, 128, 3))
    class_array = np.ndarray(shape=(0, 1000))
    start = time.time()
    for image_batch, image_labels in tqdm(dataset, total=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // 32):
        img_array = np.concatenate((img_array, image_batch))
        class_array = np.concatenate((class_array, image_labels))

        # print(image_batch.shape)
        # print(image_labels.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Snake Making GAN.')
    parser.add_argument(
        '--final-img-size',
        help='Set the final img size',
        type=int
    )
    args = parser.parse_args()
    main(
        final_img_size=args.final_img_size
    )