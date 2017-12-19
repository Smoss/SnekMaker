import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Input, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, UpSampling1D
from keras.layers.pooling import GlobalAveragePooling2D
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import Sequential, load_model, Model
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
import numpy as np
from keras import optimizers
from pprint import pprint

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 69988
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 833  
num_preprocess_threads = 16
batch_size = 8
epochs = 60
img_size = 139
num_samples = 4096 * 16
num_val_samples = 4096 * 2
try_num = 2
directory = 'C:\\Users\\User\\Documents\\SnekMaker\\Sneks\\Snakes\\'
num_params = 36
try_num = 5
inc_name = 'Inception'
num_layers = 25
model_name = inc_name + '_GeneratorPlus' + str(try_num) + '_' + str(num_layers)
generator_name = "SnekGenerator" + str(try_num)

def main(unused_argv):
	pprint(model_name)
	sess = tf.Session()
	K.set_session(sess)
	model = load_model(model_name + ".hdf5")
	model = Model(model.input, model.layers[-2].output)
	pprint(len(model.layers))
	model.save(generator_name + ".hdf5")
	sneks = model.predict(np.random.random_sample((1024, num_params)))
	pprint(sneks.shape)
	x = 0
	sneks = sneks * 255
	for snek in sneks.astype(int):
		cv2.imwrite(directory + generator_name + "_" + str(x) + ".png",snek)
		x+=1
if __name__ == "__main__":
  tf.app.run()