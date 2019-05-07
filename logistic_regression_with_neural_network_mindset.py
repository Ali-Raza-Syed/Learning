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

#%% Function for training on multiple images
# inputs:
    # -images --> images to fit model to, of shape (number of images, height, width, number of channels)
    # -labels --> images labels of shape (1, number of images)
    # -epochs --> number of iterations for updating parameters, an integer
    # -learning_rate --> learning rate for updation of weights and bias
    # -seed --> seed value for random function. Default is None (for random), an integer
    # -print_loss_flag --> whether to print loss or not, bool
    # -print_after_epochs --> number of epochs after which to print the loss
# outputs:
    # -parameters_dictionary --> final parameters dictionary after model fitting,
    #                            {'weights': weights matrix of shape (number of image nodes or pixels, 1),
    #                            'bias': bias, a float}    
def train_on_multiple_images(images, labels, epochs=1000, learning_rate=0.01, seed=None, print_loss_flag=False, print_after_epochs=100):
    num_of_weights = images.shape[1] * images.shape[2] * images.shape[3]
    np.random.seed(seed)
    weights = np.random.uniform(low=-1, high=1, size=(num_of_weights, 1)) * 1
    bias = np.random.rand() * 1
    parameters_dictionary = {'weights': weights, 'bias': bias}
    num_of_images = images.shape[0]
    for epoch_index in range(epochs):
        averaged_derivatives_dictionary = {'dw': np.zeros([num_of_weights, 1]), 'db': 0}
        for image_index in range(num_of_images):
            current_image = images[image_index]
            network_output = logistic_unit_output(current_image, parameters_dictionary)    
            derivatives_dictionary = calculate_derivatives(current_image, network_output, labels[:, image_index])
            averaged_derivatives_dictionary['dw'] += derivatives_dictionary['dw']
            averaged_derivatives_dictionary['db'] += derivatives_dictionary['db']
        
        averaged_derivatives_dictionary['dw'] /= num_of_images
        averaged_derivatives_dictionary['db'] /= num_of_images
        parameters_dictionary = update_parameters(parameters_dictionary, averaged_derivatives_dictionary, learning_rate)
        if np.mod(epoch_index, print_after_epochs) == 0 and print_loss_flag:
            print('epoch = ', epoch_index, '\nLoss = ', calculate_loss_multiple_images(images, parameters_dictionary, labels))
            
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
    # -threshold_flag --> whether to apply threshold on calculated derivative values or not, a bool
    # -threshold --> threshold to be applied on calculated derivative values
# output: 
    # -derivatives dictionary, {'dw': derivatives of Loss w.r.t weights, of shape (number of weights, 1),
    #                          'db': derivative of Loss w.r.t bias, a float}
    
def calculate_derivatives(network_input, network_output, true_output, threshold_flag=False, threshold=1e-1):
    # derivative of Loss w.r.t z, a number
    dz = network_output - np.squeeze(true_output)

    linearized_network_input = linearize_image(network_input)
    
    #derivatives of Loss w.r.t weights
    dw = linearized_network_input * dz
    
    #derivate of Loss w.r.t bias
    db = dz
    
    #thresholding
    if threshold_flag:
        dw[np.abs(dw) > threshold] = 0
        if db > threshold:
            db = 0
    
    derivatives_dictionary = {'dw': dw, 'db': db}
    return derivatives_dictionary

#%% Loss function implementation
# input:
    # network_output --> network output
    # true_output --> actual output
# output: loss, a float
def calculate_loss_multiple_images(images, parameters_dictionary, labels):
    loss = 0
    num_of_images = images.shape[0]
    for image_index in range(num_of_images):
        network_output = logistic_unit_output(images[image_index], parameters_dictionary)
        network_output = 1e-5 if network_output == 0 else network_output
        network_output = 1-1e-5 if network_output == 1 else network_output
        true_output = np.squeeze(labels[:, image_index])
        loss += -((true_output * np.log(network_output)) + ((1 - true_output) * np.log(1 - network_output)))
        
    return loss/num_of_images

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
    
    thresholded_output = 1 if output >= 0.5 else 0
    
    return thresholded_output

#%%Function to test model on  multiple images
# input:
    # -images --> images of shape (image number, height, width, num_of_channels)
    # -labels --> image labels of shape (1, number of images)
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, 1), 
    #                                              'bias': bias, a number}
    
# output:
    # -logistic unit output, a number
def test_model_multiple_images(images, labels, parameters_dictionary):
    num_of_images = images.shape[0]
    correct_predictions = 0
    for image_index in range(num_of_images):
        current_image = images[image_index]
        current_image_label = labels[:, image_index]
        network_output = logistic_unit_output(current_image, parameters_dictionary)
        network_output = 1 if network_output >= 0.5 else 0
        if network_output == current_image_label:
            correct_predictions += 1
            
    accuracy = correct_predictions/num_of_images
    return accuracy * 100
            

#%% Main Code
    
#%% Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#%% Example of a picture

index = 13
print('shape of train_set_x_orig: ', np.shape(train_set_x_orig))
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

image = train_set_x_orig[index]

parameters_dictionary = train_on_multiple_images(train_set_x_orig, train_set_y, epochs=1000, learning_rate=1e-2, seed=1, print_loss_flag=True)

model_accuracy = test_model_multiple_images(train_set_x_orig, train_set_y, parameters_dictionary)
print('Trained model accuracy on training set: ', model_accuracy)

model_accuracy = test_model_multiple_images(test_set_x_orig, test_set_y, parameters_dictionary)
print('Trained model accuracy on test set: ', model_accuracy)