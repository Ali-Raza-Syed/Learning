# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:31:51 2019

@author: Syed_Ali_Raza
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#%% Function for predicting correctly a single image
# inputs:
    # -image --> image to fit model to, of shape (height, width, number of channels)
    # -epochs --> number of iterations for updating parameters, an integer
    # -learning_rate --> learning rate for updation of weights and bias
    # -seed --> seed value for random function. Default is None (for random), an integer
    # -true_output --> true label of image, an integer either 0 or 1
# outputs:
    # -parameters_dictionary --> final parameters dictionary after model fitting,
    #                            {'weights': weights matrix of shape (number of image nodes or pixels, 1),
    #                            'bias': bias, a float}
def train_on_single_image(image, epochs=1000, learning_rate=0.01, seed=None, true_output=1):
    linearized_image = linearize_image(image)
    num_of_weights = linearized_image.shape[0]
    
    np.random.seed(seed)
    weights = np.random.uniform(low=-1, high=1, size=(num_of_weights, 1)) * 0.001
    bias = np.random.rand() * 0.001
    parameters_dictionary = {'weights': weights, 'bias': bias}
    
    for i in range(epochs):
        network_output = logistic_unit_output(image, parameters_dictionary)
        derivatives_dictionary = calculate_derivatives(image, network_output, true_output)
        parameters_dictionary = update_parameters(parameters_dictionary, derivatives_dictionary, learning_rate)
        if np.mod(i, 100) == 0:
            print('epoch = ', i, ' Network Output =', network_output, ', Loss = ', calculate_loss(network_output, true_output))
            
    return parameters_dictionary

#%%Function to update parameters (weights and bias)
# input:
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, 1), 
    #                                              'bias': bias, a number}
    # -derivatives_dictionary --> dictionary having {'dw': derivative of Loss w.r.t weights, of shape (number of weights, 1),
    #                                               'db': derivative of Loss w.r.t bias, a number}
    # -learning_rate --> learning rate for updation of parameters
# output:
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, 1), 
    #                                              'bias': bias, a number}
def update_parameters(parameters_dictionary, derivatives_dictionary, learning_rate):
    weights = parameters_dictionary['weights']
    bias = parameters_dictionary['bias']
    dw = derivatives_dictionary['dw']
    db = derivatives_dictionary['db']
    
    #updating parameters
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db
    
    parameters_dictionary['weights'] = weights
    parameters_dictionary['bias'] = bias
    
    return parameters_dictionary

#%% Function to calculate derivates for back propagation
#input:
    # -network_input --> network input of shape (height, width, num_of_channels)
    # -network_output --> output of network, a float
    # -true_output --> actuaL output, a float
# output: 
    # -derivatives dictionary, {'dw': derivatives of Loss w.r.t weights, of shape (number of weights, 1),
    #                          'db': derivative of Loss w.r.t bias, a float}
    
def calculate_derivatives(network_input, network_output, true_output):
    # derivative of Loss w.r.t z, a number
    dz = network_output - true_output

    linearized_network_input = linearize_image(network_input)
    
    #derivatives of Loss w.r.t weights
    dw = linearized_network_input * dz
    
    #derivate of Loss w.r.t bias
    db = dz
    
    dw[np.abs(dw) > 1e-1] = 0
    if db > 1e-1:
        db = 0
    
    derivatives_dictionary = {'dw': dw, 'db': db}
    return derivatives_dictionary

#%% Loss function implementation
# network_output --> network output
# true_output --> actual output
def calculate_loss(network_output, true_output):
    return -((true_output * np.log(network_output)) + ((1 - true_output) * np.log(1 - network_output)))

#%% Sigmoid function implementation
# x --> number to calculare sigmoid of
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#%% Function to linearize a 3-D image
# input:
    # image --> image of shape (height, width, num_of_channels) 
# output: linear image of shape (height * width * num_of_channels)
def linearize_image(image):
    # Extract dimensions of input image
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_channels= image.shape[2]
    
    # Linearize image
    linear_image = image.reshape([image_height * image_width * image_channels])
    linear_image = np.expand_dims(linear_image, axis=1)
    return linear_image
    
#%% Function for calculating output of Logistic Regression unit
# input:
    # -image --> image of shape (height, width, num_of_channels)
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, 1), 
    #                                              'bias': bias, a number}
    
# output:
    # -logistic unit output, a number
def logistic_unit_output(image, parameters_dictionary):
    weights = parameters_dictionary['weights']
    bias = parameters_dictionary['bias']
    
    linear_image = linearize_image(image)

    # Weights transposed for multiplication
    # (number_of_weights, 1) --> (1, number_of_weights)
    weights_transposed = weights.T
    
    # Multiply weights with input and add bias
    # Note that bias is a float but gets broadcasted to a higher dimensional matrix for addition
    output = np.matmul(weights_transposed, linear_image) + bias
    
    # Squeezing output of shape (1, 1) to a single float number
    output = np.squeeze(output)
    
    # Take sigmoid of output
    sigmoid_output = sigmoid(output)
    
    return sigmoid_output


#%% Main Code
    
#%% Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#%% Example of a picture
index = 13
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

image = train_set_x_orig[index]

train_on_single_image(image, epochs=10000, learning_rate=1e-2, seed=1, true_output=1)