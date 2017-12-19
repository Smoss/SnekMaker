import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Input, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, UpSampling1D
from keras.layers.pooling import GlobalAveragePooling2D
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import Sequential, load_model, Model
from keras.layers.normalization import BatchNormalization
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
epochs = 5
img_size = 139
num_samples = 4096 * 2
num_val_samples = 4096 // 8
try_num = 2
directory = 'C:\\Users\\User\\Documents\\CheckSneks'
num_pics = len(os.listdir(directory + "\\Snakes"))
num_params = 36
try_num = 5
inc_name = 'Inception'
num_datasets = 60

def main(unused_argv):
	sess = tf.Session()
	K.set_session(sess)
	main_input = Input(shape=(num_params,), dtype='float32', name='main_input')
	x = Dense(192*num_params*4, activation='relu')(main_input)
	x = BatchNormalization(momentum=0.9)(x)
	x = Dropout(.4)(x)	
	x = Reshape((12, 12, 192))(x)
	x = Conv2DTranspose(128, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Conv2DTranspose(128, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = UpSampling2D(size = (2,2))(x)
	x = Conv2DTranspose(64, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Conv2DTranspose(64, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = UpSampling2D(size = (2,2))(x)
	x = Conv2DTranspose(32, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Conv2DTranspose(32, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = UpSampling2D(size = (2,2))(x)
	x = Conv2DTranspose(16, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Conv2DTranspose(16, kernel_size = (3,3), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	New_img = Conv2DTranspose(3, kernel_size = (3,3), activation = 'relu')(x)
	temp_model = Model(main_input, New_img)
	pprint(temp_model.predict(np.random.random_sample((1,num_params))).shape)
	num_layers = len(temp_model.layers)
	model_name = inc_name + '_GeneratorPlus' + str(try_num) + '_' + str(num_layers)
	pre_name = 'Pre' + model_name
	model = Model(inputs = main_input, outputs = New_img)
	incModel = load_model(inc_name + "Wunderkid2.hdf5")
	for layer in incModel.layers:
		layer.trainable = False
	genModel = incModel(New_img)
	genModel = Model(inputs = main_input, outputs = genModel)
	print(len(model.layers))
	for layer in genModel.layers[len(model.layers):]:
		layer.trainable = False 
	genModel.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	for x in range(0, num_datasets):
		input_values = np.random.random_sample((num_samples, num_params))
		validation_data = np.random.random_sample((num_val_samples, num_params))
		genModel.fit(input_values, np.zeros(num_samples),
			validation_data = (validation_data, np.zeros(num_val_samples)),
			epochs = int(epochs * (x+1)),
			batch_size = batch_size,
			callbacks=[TensorBoard(log_dir='SnekChecker' + model_name), ModelCheckpoint(model_name + '.hdf5')],
			initial_epoch = int(x * epochs))
	genModel.save(model_name + '.hdf5')
if __name__ == "__main__":
  tf.app.run()