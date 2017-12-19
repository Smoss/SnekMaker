import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import Sequential, load_model, Model
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers
from keras.applications.vgg19 import VGG19
import numpy as np
from keras import optimizers
from pprint import pprint

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 69988
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 833  
num_preprocess_threads = 16
batch_size = 16
epochs = 200
img_size = 128
model_name = 'Wunderkid'
directory = 'C:\\Users\\User\\Documents\\CheckSneks'
num_pics = len(os.listdir(directory + "\\Snakes"))


def main(unused_argv):
	# build the VGG16 network
	# model = VGG19(weights='imagenet', include_top=False, 
	# 	input_shape = (img_size, img_size, 3))
	model = load_model("PreWunderkid.hdf5")
	print('Model loaded.')
	# # build a classifier model to put on top of the convolutional model
	# top_model = Flatten()(model.output)
	# top_model = Dense(256, activation='relu')(top_model)
	# top_model = Dropout(0.5)(top_model)
	# top_model = Dense(1, activation='sigmoid')(top_model)

	# #model =  Model(inputs=[main_input], outputs=[top_model])
	# # set the first 25 layers (up to the last conv block)
	# # to non-trainable (weights will not be updated)
	# for layer in model.layers[:-4]:
	# 	layer.trainable = False

	# # add the model on top of the convolutional base
	# model = Model(inputs = model.input, outputs=top_model)

	# # compile the model with a SGD/momentum optimizer
	# # and a very slow learning rate.
	# model.compile(loss='binary_crossentropy',
	#               optimizer='adadelta',
	#               metrics=['accuracy'])

	# prepare data augmentation configuration
	train_datagen = ImageDataGenerator(
	    rescale=1. / 255,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
		'C:\\Users\\User\\Documents\\FirstRoundTraining',
	    target_size=(img_size, img_size),
	    batch_size=batch_size,
	    class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(
		'C:\\Users\\User\\Documents\\SecondRoundTraining',
	    target_size=(img_size, img_size),
	    batch_size=batch_size,
	    class_mode='binary')

	# # fine-tune the model
	# model.fit_generator(
	#     train_generator,
	#     steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size,
	#     epochs=epochs/40,
	#     validation_data=validation_generator,
	#     validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // batch_size,
	# 	callbacks=[TensorBoard(log_dir='SnekCheckerPre' + model_name), ModelCheckpoint('Pre' + model_name + '.hdf5', save_best_only=True, mode = 'min')])
	# print("Trained Top")
	# for layer in model.layers[-9:]:
	# 	layer.trainable = True

	model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',
	               metrics=['accuracy'])
	# fine-tune the model
	model.fit_generator(
	    train_generator,
	    steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size,
	    epochs=epochs,
	    validation_data=validation_generator,
	    validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // batch_size,
		callbacks=[TensorBoard(log_dir='SnekChecker' + model_name), ModelCheckpoint(model_name + '.hdf5', save_best_only=True, mode = 'min')])

if __name__ == "__main__":
  tf.app.run()