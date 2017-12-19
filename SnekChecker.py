import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import Sequential, load_model
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000  
num_preprocess_threads = 16
batch_size = 32
epochs = 60
img_size = 158
model_num = "Inception"
directory = 'C:\\Users\\User\\Documents\\SnekMaker\\Sneks'
num_pics = len(os.listdir(directory + "\\Snakes"))

def main(unused_argv):

	sess = tf.Session()
	K.set_session(sess)
	model = load_model("InceptionWunderkid2.hdf5")
	test_datagen = ImageDataGenerator(rescale=1./255)
	train_generator = test_datagen.flow_from_directory(
	        directory,  # this is the target directory
	        target_size=(img_size, img_size),  # all images will be resized to 150x150
	        batch_size=batch_size,
	        shuffle = False,
	        class_mode='binary') 
	val = model.predict_generator(
	        train_generator,
	        steps = num_pics // batch_size + 1,
	        verbose = 1)
	files = sorted(os.listdir(directory + "\\Snakes"))
	val = zip(files, val[:num_pics])
	with open(model_num + "_predictions.txt", 'w+') as f:
		[f.write(str(x) + ':' + str(y) + '\n') for x,y in val]

if __name__ == "__main__":
  tf.app.run()