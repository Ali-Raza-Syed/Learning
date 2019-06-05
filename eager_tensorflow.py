from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from lr_utils import load_dataset

def linearize_images(images):
    # Extract dimensions of input image
    num_of_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    image_channels = images.shape[3]
    
    linearized_images = images.reshape([num_of_images, image_height * image_width * image_channels])
#    linearized_images = linearized_images.T
    
    return linearized_images

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

activation = 'LeakyReLU'
prediction_threshold = 0.5
learning_rate = 1e-2
epochs = 1000
seed = 1
num_of_nodes = 1
num_images = train_set_x_orig.shape[0]
num_input_features = train_set_x_orig.shape[1] * train_set_x_orig.shape[2] * train_set_x_orig.shape[3]

np.random.seed(seed)
W = np.random.uniform(low=-1, high=1, size=(num_input_features, num_of_nodes)) * 1e-5  
B = np.random.uniform(low=-1, high=1, size=(num_of_nodes, 1)) * 1e-5

W = tf.Variable(W, tf.float64)
B = tf.Variable(B, tf.float64)

dataset = linearize_images(train_set_x_orig)
labels = train_set_y.T

dataset = tf.data.Dataset.from_tensors((dataset, labels))

for inputs, labels in dataset.take(-1):
    
    bce = tf.keras.losses.BinaryCrossentropy()
    with tf.GradientTape() as tape:
        
        loss = bce(tf.cast(A, tf.float64), tf.cast(labels, tf.float64))
        print('loss = ', loss)
        grad = tape.gradient(loss, B)
        print('grad = ', grad)

#w = tf.Variable([[2.0]])
#with tf.GradientTape() as tape:
#  loss = w * w * w + w * w
#
#grad = tape.gradient(loss, w)
#grad = tape.gradient(w, w)
#print(grad)