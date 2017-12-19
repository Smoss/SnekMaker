from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import os

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)
snake_val = 1
not_snake_val = 0
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 32, 32, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[7, 7],
      padding="same",
      activation=tf.nn.relu)

 # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def convert_Folder_To_Array(folder, label):
	files = [folder + "\\" + x for x in os.listdir(folder)]
	labels = tf.constant(label, shape=[len(files)])
	return files, labels, len(files)

def input_fn():
  num_preprocess_threads = 16
  batch_size = 128
  min_queue_examples = batch_size * .4
  train_data_snakes, snake_labels, num_snakes= convert_Folder_To_Array("C:\\Users\\User\\Documents\\SnakeSquares", snake_val)
  print("Made Sneks")
  train_data_not_snakes, not_snake_labels, num_not_snakes = convert_Folder_To_Array("C:\\Users\\User\\Documents\\cifar-10-batches-py\\Initial\\Squares", not_snake_val)
  print("Made not Sneks")
  train_data = tf.train.string_input_producer(train_data_snakes + train_data_not_snakes)
  train_names = train_data_snakes + train_data_not_snakes
  reader = tf.WholeFileReader()
  key, value = reader.read(train_data)
  my_img = tf.image.decode_png(value, channels=3)
  init_op = tf.global_variables_initializer()
  train_labels = tf.cast(tf.concat([snake_labels, not_snake_labels], axis = 0), tf.int32)
  print(str(train_labels))
  # images, label_batch = tf.train.shuffle_batch(
  #   [my_img, train_labels],
  #   batch_size=batch_size,
  #   num_threads=num_preprocess_threads,
  #   capacity=min_queue_examples + 3 * batch_size,
  #   min_after_dequeue=min_queue_examples)
  with tf.Session() as sess:
	  sess.run(init_op)

	  # Start populating the filename queue.

	  coord = tf.train.Coordinator()
	  threads = tf.train.start_queue_runners(coord=coord)
	  print("supply side jesus")
	  imgs = []
	  for i in range(num_snakes + num_not_snakes):
	  	val = tf.cast(my_img.eval(), tf.float32)
	  	val = val / 255.0
	  	print(val)
	  	imgs.append(val)
	  print("is back from the gates of hell")


	  coord.request_stop()
	  coord.join(threads)
	  imgs = tf.convert_to_tensor(imgs) 
	  print(imgs)
  return imgs, train_labels



# Our application logic will be added here
def main(unused_argv):
  # Load training and eval data
  snek_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="snek_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  # Train the model
  snek_classifier.fit(
  	  input_fn = input_fn,
      steps=10000,
      monitors=[logging_hook])


if __name__ == "__main__":
  tf.app.run()