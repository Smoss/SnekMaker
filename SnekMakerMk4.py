import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Input, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, UpSampling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import Sequential, Model
import cv2
import os
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
import numpy as np
from pprint import pprint

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5032 * 2
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1008  
num_preprocess_threads = 16
batch_size = 32
epochs = 1
img_size = 223
num_params = 100
first_layer_dim = 20
num_samples = 5032
num_val_samples = 0
gen_ratio = 1
color_channels = 3
directory = 'C:\\Users\\User\\Documents\\CheckSneks'
num_pics = len(os.listdir(directory + "\\Snakes"))

def main(unused_argv):

	sess = tf.Session()
	K.set_session(sess)
	main_input = Input(shape=(num_params,), dtype='float32', name='main_input')
	x = Dense(128*first_layer_dim*first_layer_dim, activation='relu')(main_input)
	#x = Dropout(.4)(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Reshape((first_layer_dim, first_layer_dim, 128))(x)
	x = Conv2DTranspose(256, kernel_size = 5, strides = (2,2), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	x = Conv2DTranspose(128, kernel_size = 5, strides = (2,2), activation = 'relu')(x)
	x = BatchNormalization(momentum=0.9)(x)
	New_img = Conv2DTranspose(color_channels, kernel_size = (5,5), activation = 'sigmoid')(x)
	gen_model = Model(main_input, New_img)
	input_values = np.random.random_sample((num_samples + num_val_samples, num_params))
	sneks = gen_model.predict(input_values)
	pprint(sneks.shape)
	img_size = sneks.shape[1]
	y = 0
	x = 0
	sneks = sneks * 255
	for y in range(0, num_samples + num_val_samples):
		if y%10 == 0:
			cv2.imwrite('C:\\Users\\User\\Documents\\SnekMaker\\Sneks\\Sneks\\Generation_' + str(x) + "_" + str(y) + ".png",sneks[y].astype(int))
		cv2.imwrite('C:\\Users\\User\\Documents\\SnekMaker\\TrainingData\\Squares\\' + str(y) + ".png",sneks[y].astype(int))
	id_model = Sequential()
	# id_model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu',))
	# #id_model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu',))
	# id_model.add(MaxPooling2D(pool_size = (2,2)))
	# #id_model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
	# id_model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
	# id_model.add(MaxPooling2D(pool_size = (2,2)))
	#id_model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	id_model.add(Conv2D(128, kernel_size = (5,5), activation = 'relu', strides = (2,2), input_shape = (img_size, img_size, color_channels)))
	id_model.add(MaxPooling2D(pool_size = (2,2)))
	#id_model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
	id_model.add(Conv2D(256, kernel_size = (5,5), activation = 'relu', strides = (2,2)))
	id_model.add(MaxPooling2D(pool_size = (2,2)))
	#print(id_model.layers[-1].output)
	#id_model.add(GlobalMaxPooling2D())
	id_model.add(Flatten())
	id_model.add(Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
	id_model.add(Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
	#id_model.add(Dropout(.35))
	id_model.add(Dense(1, activation='sigmoid'))
	id_model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
	num_layers_gen = len(gen_model.layers)
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
	id_model_num = 'GAN_1+' + str(img_size)
	pprint(id_model.output.shape)
	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)
	train_model = Sequential()
	train_model.add(gen_model)
	train_model.add(id_model)
	#print(input_fn())
	# this is a generator that will read pictures found in
	# subfolders of 'data/train', and indefinitely generate
	# batches of augmented image data
	first_run = True
	for x in range(0,1000):
		train_generator = train_datagen.flow_from_directory(
		        'C:\\Users\\User\\Documents\\SnekMaker\\TrainingData',  # this is the target directory
		        target_size=(img_size, img_size),  # all images will be resized to 128x128
		        batch_size=batch_size,
		        classes = ['Snakes', 'Squares'],
		        class_mode='binary')  # since we use binary_crossentropy loss, we need binary label
		for layer in train_model.layers[0].layers:
			layer.trainable = False
		for layer in train_model.layers[-1].layers:
			layer.trainable = True
		num_layers_id = len(id_model.layers)
		id_model.fit_generator(
		        train_generator,
		        steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size,
		        epochs=int(epochs * (x+1)),
		        callbacks=[TensorBoard(log_dir='SnekGANChecker' + id_model_num), ModelCheckpoint(id_model_num + 'CheckerTry.hdf5', save_best_only=False, mode = 'min')],
		        initial_epoch = int(x * epochs))
		pprint(train_model.predict(input_values[0:1])[0])
		for layer in train_model.layers[0].layers:
			layer.trainable = True
		for layer in train_model.layers[-1].layers:
			layer.trainable = False
		if first_run:
			train_model.compile(loss='binary_crossentropy',
		              optimizer='Adam',
		              metrics=['accuracy'])
			first_run = False
		input_values = np.random.random_sample((num_samples, num_params))
		validation_data = np.random.random_sample((num_val_samples, num_params))
		train_model.fit(input_values, np.zeros(num_samples),
				epochs = int((epochs * gen_ratio) * (x+1)),
				batch_size = batch_size//4,
		        callbacks=[TensorBoard(log_dir='SnekMaker' + id_model_num), ModelCheckpoint(id_model_num + 'GeneratorTry.hdf5')],
		        initial_epoch = int(x * (epochs*gen_ratio)))
		input_values = np.random.random_sample((num_samples + num_val_samples, num_params))
		sneks = gen_model.predict(input_values)
		y = 0
		sneks = sneks * 255
		for y in range(0, num_samples + num_val_samples):
			if y%10 == 0:
				cv2.imwrite('C:\\Users\\User\\Documents\\SnekMaker\\Sneks\\Sneks\\Generation_' + str(x) + "_" + str(y) + ".png",sneks[y].astype(int))
			cv2.imwrite('C:\\Users\\User\\Documents\\SnekMaker\\TrainingData\\Squares\\' + str(y) + ".png",sneks[y].astype(int))

	files = sorted(os.listdir(directory + "\\Snakes"))
	val = zip(files, val[:num_pics])
	with open(id_model_num + "_predictions.txt", 'w+') as f:
		[f.write(str(x) + ':' + str(y) + '\n') for x,y in val]

if __name__ == "__main__":
  tf.app.run()