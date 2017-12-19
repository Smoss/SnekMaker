import tensorflow as tf
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalMaxPooling2D
from keras import backend as K
from keras.objectives import categorical_crossentropy
from keras.models import Sequential
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import regularizers

snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 71686
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1183  
num_preprocess_threads = 16
batch_size = 32
epochs = 200
img_size = 256
model_num = '28+' + str(img_size)
directory = 'C:\\Users\\User\\Documents\\CheckSneks'
num_pics = len(os.listdir(directory + "\\Snakes"))
# def read_my_file_format(filename_queue):
#   reader = tf.SomeReader()
#   key, record_string = reader.read(filename_queue)
#   example, label = tf.decode_image(record_string)
#   processed_example = some_processing(example)
#   return processed_example, label

# def input_pipeline(filenames, batch_size, num_epochs=None):
#   filename_queue = tf.train.string_input_producer(
#       filenames, num_epochs=num_epochs, shuffle=True)
#   example, label = read_my_file_format(filename_queue)
#   # min_after_dequeue defines how big a buffer we will randomly sample
#   #   from -- bigger means better shuffling but slower start up and more
#   #   memory used.
#   # capacity must be larger than min_after_dequeue and the amount larger
#   #   determines the maximum we will prefetch.  Recommendation:
#   #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
#   min_after_dequeue = 10000
#   capacity = min_after_dequeue + 3 * batch_size
#   example_batch, label_batch = tf.train.shuffle_batch(
#       [example, label], batch_size=batch_size, capacity=capacity,
#       min_after_dequeue=min_after_dequeue)
#   return example_batch, label_batch

# def convert_Folder_To_Array(folder, label):
# 	files = [folder + "\\" + x for x in os.listdir(folder)]
# 	labels = tf.constant(label, shape=[len(files)])
# 	return files, labels, len(files)

# def input_fn():

#   min_queue_examples = batch_size * .4
#   train_data_snakes, snake_labels, num_snakes= convert_Folder_To_Array("C:\\Users\\User\\Documents\\SnakeSquares", snake_val)
#   print("Made Sneks")
#   train_data_not_snakes, not_snake_labels, num_not_snakes = convert_Folder_To_Array("C:\\Users\\User\\Documents\\cifar-10-batches-py\\Initial\\Squares", not_snake_val)
#   print("Made not Sneks")
#   train_data = tf.train.string_input_producer(train_data_snakes + train_data_not_snakes)
#   train_names = train_data_snakes + train_data_not_snakes
#   reader = tf.WholeFileReader()
#   key, value = reader.read(train_data)
#   my_img = tf.image.decode_image(value, channels=3)
#   my_img = tf.reshape(my_img, [3, 32, 32])
#   my_img = tf.cast(my_img, tf.float32)
#   my_img = tf.image.per_image_standardization(my_img)
#   init_op = tf.global_variables_initializer()
#   train_labels = tf.cast(tf.concat([snake_labels, not_snake_labels], axis = 0), tf.int32)
#   print(str(my_img))
#   print(str(train_labels))
#   images, label_batch = tf.train.shuffle_batch(
#     [my_img, train_labels],
#     batch_size=batch_size, 
#     capacity=min_queue_examples + 3 * batch_size,
#     min_after_dequeue=min_queue_examples)
#   # with tf.Session() as sess:
# 	 #  sess.run(init_op)

# 	 #  # Start populating the filename queue.

# 	 #  coord = tf.train.Coordinator()
# 	 #  threads = tf.train.start_queue_runners(coord=coord)
# 	 #  print("supply side jesus")
# 	 #  imgs = []
# 	 #  for i in range(num_snakes + num_not_snakes):
# 	 #  	val = tf.cast(my_img.eval(), tf.float32)
# 	 #  	val = val / 255.0
# 	 #  	print(val)
# 	 #  	imgs.append(val)
# 	 #  print("is back from the gates of hell")


# 	 #  coord.request_stop()
# 	 #  coord.join(threads)
# 	 #  imgs = tf.convert_to_tensor(imgs) 
# 	 #  print(imgs)
#   return imgs, train_labels

def main(unused_argv):

	sess = tf.Session()
	K.set_session(sess)

	model = Sequential()
	# Keras layers can be called on TensorFlow tensors:
	model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu', input_shape = (img_size, img_size, 3)))
	#model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu',))
	model.add(MaxPooling2D(pool_size = (2,2)))
	#model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
	model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	#model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	#model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
	model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
	#print(model.layers[-1].output)
	model.add(GlobalMaxPooling2D())
	model.add(Dropout(.4))
	model.add(Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(.4))
	model.add(Dense(64, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dropout(.4))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

	train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)
	#print(input_fn())
	# this is a generator that will read pictures found in
	# subfolders of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
	        'C:\\Users\\User\\Documents\\FirstRoundTraining',  # this is the target directory
	        target_size=(img_size, img_size),  # all images will be resized to 128x128
	        batch_size=batch_size,
	        class_mode='binary')  # since we use binary_crossentropy loss, we need binary label
	validation_generator = test_datagen.flow_from_directory(
	        'C:\\Users\\User\\Documents\\SecondRoundTraining',
	        target_size=(img_size, img_size),
	        batch_size=batch_size,
	        class_mode='binary')
	val = model.fit_generator(
	        train_generator,
	        steps_per_epoch=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // batch_size,
	        epochs=epochs,
	        validation_data=validation_generator,
	        validation_steps=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL // batch_size,
	        callbacks=[TensorBoard(log_dir='SnekChecker' + model_num), ModelCheckpoint(model_num + 'try.hdf5', save_best_only=True, mode = 'min')])
	predict_datagen = ImageDataGenerator(rescale=1./255)
	predict_generator = predict_datagen.flow_from_directory(
	        'C:\\Users\\User\\Documents\\CheckSneks',  # this is the target directory
	        target_size=(img_size, img_size),  # all images will be resized to 128x128
	        batch_size=batch_size,
	        shuffle = False,
	        class_mode='binary') 
	
	val = model.predict_generator(
	        predict_generator,
	        steps = num_pics // batch_size + 1)
	files = sorted(os.listdir(directory + "\\Snakes"))
	val = zip(files, val[:num_pics])
	with open(model_num + "_predictions.txt", 'w+') as f:
		[f.write(str(x) + ':' + str(y) + '\n') for x,y in val]

if __name__ == "__main__":
  tf.app.run()