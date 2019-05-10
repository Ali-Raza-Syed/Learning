# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:59:30 2019

@author: Syed_Ali_Raza
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import warnings
warnings.filterwarnings("error")

#%% Function for training on multiple images
# inputs:
    # -images --> images to fit model to, of shape (number of images, height, width, number of channels)
    # -num_of_nodes --> number of nodes to be made in a single layer
    # -labels --> images labels of shape (1, number of images)
    # -epochs --> number of iterations for updating parameters, an integer
    # -learning_rate --> learning rate for updation of weights and bias
    # -seed --> seed value for random function. Default is None (for random), an integer
    # -print_loss_flag --> whether to print loss or not, bool
    # -print_after_epochs --> number of epochs after which to print the loss
# outputs:
    # -parameters_dictionary --> final parameters dictionary after model fitting,
    #                            {'weights': weights matrix of shape (number of image nodes or pixels, number of nodes in first layer),
    #                            'bias': bias, matrix of shape (number of nodes in first layer, 1),
    #                            'z': useless as an output from here. It was meant for using in calculating derivatives}    
def train_on_multiple_images(images, num_of_nodes, labels, activation='ReLU', epochs=1000, learning_rate=0.01, seed=None, print_loss_flag=False, print_after_epochs=100):
    num_of_weights = images.shape[1] * images.shape[2] * images.shape[3]
    np.random.seed(seed)
    weights = np.random.uniform(low=-1, high=1, size=(num_of_weights, num_of_nodes)) * 1e-5
    bias = np.random.uniform(low=-1, high=1, size=(num_of_nodes, 1)) * 1e-5
    parameters_dictionary = {'weights': weights, 'bias': bias}
    num_of_images = images.shape[0]
    for epoch_index in range(epochs):
        averaged_derivatives_dictionary = {'dw': np.zeros([num_of_weights, num_of_nodes]), 'db': np.zeros([num_of_nodes, 1])}
        Zs = []
        for image_index in range(num_of_images):
            current_image = images[image_index]
            network_output = logistic_unit_output(image=current_image, parameters_dictionary=parameters_dictionary, activation=activation)
            
            Zs.append(parameters_dictionary['z'])
            
            derivatives_dictionary = calculate_derivatives(network_input=current_image, network_output=network_output, true_output=labels[:, image_index], activation=activation, parameters_dictionary=parameters_dictionary, threshold_flag=False)
            averaged_derivatives_dictionary['dw'] += derivatives_dictionary['dw']
            averaged_derivatives_dictionary['db'] += derivatives_dictionary['db']
        averaged_derivatives_dictionary['dw'] /= num_of_images
        averaged_derivatives_dictionary['db'] /= num_of_images
        
        derivatives_dictionary_averaged = averaged_derivatives_dictionary['dw']
        
        
        parameters_dictionary = update_parameters(parameters_dictionary, averaged_derivatives_dictionary, learning_rate)
        if np.mod(epoch_index, print_after_epochs) == 0 and print_loss_flag:
            print('epoch = ', epoch_index, '\nLoss = ', calculate_loss_multiple_images(images, parameters_dictionary, labels))
            np.shape(weights[weights < 0])
    return parameters_dictionary

#%%Function to update parameters (weights and bias)
# input:
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, number of nodes in the layer), 
    #                                              'bias': biases of shape (number of nodes in the layer, 1)}
    # -derivatives_dictionary --> dictionary having {'dw': derivative of Loss w.r.t weights, of shape (number of weights, number of nodes in the layer),
    #                                               'db': derivative of Loss w.r.t biases of shape (number of nodes in the layer, 1)}
    # -learning_rate --> learning rate for updation of parameters
# output:
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, number of nodes in the layer), 
    #                                              'bias': biases of shape (number of nodes in the layer, 1)}
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
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of nodes/pixels in image, number of nodes in the layer), 
    #                                              'bias': bias, a matrix of shape (number of nodes in the layer, 1),
    #                                              'z': output of individual nodes before activation. Will be used in calculating derivatives}
# output: 
    # -derivatives dictionary having {'dw': derivative of Loss w.r.t weights, of shape (number of weights, number of nodes in the layer),
    #                                'db': derivative of Loss w.r.t biases of shape (number of nodes in the layer, 1)}
def calculate_derivatives(network_input, network_output, true_output, parameters_dictionary, activation='ReLU', threshold_flag=False, threshold=1e-1):
    
    z = parameters_dictionary['z']
    
    # derivative of Loss w.r.t z, a number
    dz = network_output - np.squeeze(true_output)

    linearized_network_input = linearize_image(network_input)
    
    num_of_input_nodes = np.shape(linearized_network_input)[0]
    
    #derivative of activation function w.r.t z
    #shape will be (number of nodes in layer, 1)
    activation_derivative = np.copy(z)
    if activation == 'LeakyReLU':
        activation_derivative[activation_derivative >= 0] = 1
        activation_derivative[activation_derivative < 0] = 0.1
    elif activation == 'Sigmoid':
        activation_derivative = sigmoid(activation_derivative) * (1 - sigmoid(activation_derivative))
    elif activation == 'ReLU':
        activation_derivative[activation_derivative >= 0] = 1
        activation_derivative[activation_derivative < 0] = 0
    
    #for calculating derivatives for all the wieghts
    #shape will be (number of nodes in layer, number of input nodes)
    activation_derivative_repeated = np.repeat(activation_derivative, num_of_input_nodes, axis=1)
    
    #for matching the rows with rows of input
    #shape will be (number of input nodes, number of nodes in layer)
    activation_derivative_repeated = activation_derivative_repeated.T
    
    #derivatives of Loss w.r.t weights
    dw = linearized_network_input * dz * activation_derivative_repeated
    
    #derivate of Loss w.r.t bias
    db = dz * activation_derivative
    
    #thresholding
    if threshold_flag:
        dw = np.clip(dw, -threshold, threshold)
        db = np.clip(db, -threshold, threshold)

    
    derivatives_dictionary = {'dw': dw, 'db': db}
    return derivatives_dictionary

#%% calculate_derivatives test function
def test_calculate_derivatives():
    #2x2x1 image
    image = np.array([[[2], [5]], [[4], [1]]])
    # 2 nodes in the layer
    weights = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    bias = np.array([[1], [2]])
#    parameters_dictionary = {'weights': weights, 'bias': bias}
#    network_output = logistic_unit_output(image=image, parameters_dictionary=parameters_dictionary, activation='sigmoid', prediction_threshold_flag=False)
    #can also come from logistic_unit_output function
    network_output = 0.88
    z = np.array([[0.5], [0.6]])
    parameters_dictionary = {'weights': weights, 'bias': bias, 'z': z}
    true_output = 1
    
    derivatives_dictionary = calculate_derivatives(network_input=image, network_output=network_output, true_output=true_output, activation='sigmoid', parameters_dictionary=parameters_dictionary, threshold_flag=False, threshold=1e-1)
    a = 1
    
#%% Loss function implementation
# input:
    # network_output --> network output
    # true_output --> actual output
# output: loss, a float
def calculate_loss_multiple_images(images, parameters_dictionary, labels):
    loss = 0
    num_of_images = images.shape[0]
    for image_index in range(num_of_images):
        network_output = logistic_unit_output(images[image_index], parameters_dictionary, activation='ReLU')
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
# output: linear image of shape (height * width * num_of_channels, 1)
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
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of nodes/pixels in image, number of nodes in the layer), 
    #                                              'bias': bias, a matrix of shape (number of nodes in the layer, 1)}
    
# output:
    # -logistic unit output, a number
def logistic_unit_output(image, parameters_dictionary, activation='ReLU'):
    weights = parameters_dictionary['weights']
    bias = parameters_dictionary['bias']
    
    linear_image = linearize_image(image)

    # Weights transposed for multiplication
    # (number of nodes/pixels in image, number of nodes in the layer) --> (number of nodes in the layer, number of nodes/pixels in image)
    weights_transposed = weights.T
    
    # Multiply weights with input and add bias
    # Note that bias is a vector but gets broadcasted to a higher dimensional matrix for addition
    nodes_outputs = np.matmul(weights_transposed, linear_image) + bias
    
    parameters_dictionary['z'] = nodes_outputs
    
    #For introducing non-linearity. If hadn't done so, all the nodes sum would be same as using a single node
    nodes_outputs_activated = np.copy(nodes_outputs)
    prediction_before_activation = 0
    if activation == 'LeakyReLU':
        for i in range(np.shape(nodes_outputs_activated)[0]):
            if nodes_outputs_activated[i, 0] < 0:
                nodes_outputs_activated[i, 0] = nodes_outputs_activated[i, 0] * 0.1
    elif activation == 'ReLU':
        nodes_outputs_activated[nodes_outputs_activated < 0] = 0
    elif activation == 'Sigmoid':
        nodes_outputs_activated = sigmoid(nodes_outputs_activated)    
        
    prediction_before_activation = np.sum(nodes_outputs_activated)
    # Take sigmoid of output
    prediction_sigmoid = sigmoid(prediction_before_activation)
    
    return prediction_sigmoid

#%% logistic_unit_output test function
def test_logistic_unit_output():
    #2x2x1 image
    image = np.array([[[2], [5]], [[4], [1]]])
    # 2 nodes in the layer
    weights = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    bias = np.array([[1], [2]])
    parameters_dictionary = {'weights': weights, 'bias': bias}
    prediction_sigmoid = logistic_unit_output(image=image, parameters_dictionary=parameters_dictionary, activation='ReLU', prediction_threshold_flag=True)
    a = 1

#%%Function to test model on  multiple images
# input:
    # -images --> images of shape (image number, height, width, num_of_channels)
    # -labels --> image labels of shape (1, number of images)
    # -parameters_dictionary --> dictionary having {'weights': weights of shape (number of weights, 1), 
    #                                              'bias': bias, a number}
    
# output:
    # -logistic unit output, a number
def test_model_multiple_images(images, labels, parameters_dictionary, activation='ReLU', prediction_threshold=0.5):
    num_of_images = images.shape[0]
    correct_predictions = 0
    #for checking all outputs
    network_outputs = []
    for image_index in range(num_of_images):
        current_image = images[image_index]
        current_image_label = labels[:, image_index]
        network_output = logistic_unit_output(image=current_image, parameters_dictionary=parameters_dictionary, activation=activation)
        network_outputs.append(network_output)
        network_output= 1 if network_output >= prediction_threshold else 0
        if network_output == current_image_label:
            correct_predictions += 1
            
    accuracy = correct_predictions/num_of_images
    return accuracy * 100, network_outputs


#%% Main Code
    
#%% Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

##%% Example of a picture
#
#index = 13
#print('shape of train_set_x_orig: ', np.shape(train_set_x_orig))
#plt.imshow(train_set_x_orig[index])
#print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
#
#image = train_set_x_orig[index]


#train_set_x_orig_min = train_set_x_orig.min(axis=(1, 2), keepdims=True)
#train_set_x_orig_max = train_set_x_orig.max(axis=(1, 2), keepdims=True)

#train_set_x_orig = (train_set_x_orig - train_set_x_orig_min)/(train_set_x_orig_max-train_set_x_orig_min)

activation = 'Sigmoid'
prediction_threshold = 0.5
parameters_dictionary = train_on_multiple_images(images=train_set_x_orig, num_of_nodes=1, labels=train_set_y, activation=activation, epochs=1000, learning_rate=1e-5, seed=1, print_loss_flag=True, print_after_epochs=1)

model_accuracy, network_outputs = test_model_multiple_images(train_set_x_orig, train_set_y, parameters_dictionary, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(images=train_set_x_orig, parameters_dictionary=parameters_dictionary, labels=train_set_y)
print('Trained model accuracy on training set: ', model_accuracy)
print('Trained model loss on training set: ', model_loss)

model_accuracy, _ = test_model_multiple_images(test_set_x_orig, test_set_y, parameters_dictionary, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(images=test_set_x_orig, parameters_dictionary=parameters_dictionary, labels=test_set_y)
print('Trained model accuracy on test set: ', model_accuracy)
print('Trained model loss on test set: ', model_loss)

#%% Logistic unit output test
#test_logistic_unit_output()

#%% calculate test_calculate_derivatives
#test_calculate_derivatives()