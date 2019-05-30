# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:59:30 2019

@author: Syed_Ali_Raza
"""
import numpy as np
from lr_utils import load_dataset
import warnings
warnings.filterwarnings("error")
import matplotlib.pyplot as plt

#%% Function for training on multiple images
# inputs:
    # -linearized_images --> images to fit model to, of shape (height*width*number of channels, number of images)
    # -layers --> number of nodes in each layer, including first and last, a row vector e.g [250, 3, 2, 1] for 250 input features, 3 and 2 hidden layer units, and 1 output layer unit
    # -labels --> images labels of shape (1, number of images)
    # -activation --> activation function being used in hidden layers
    # -epochs --> number of iterations for updating parameters, an integer
    # -learning_rate --> learning rate for updation of weights and bias
    # -seed --> seed value for random function. Default is None (for random), an integer
    # -print_loss_flag --> whether to print loss or not, bool
    # -print_after_epochs --> number of epochs after which to print the loss
    # -get_cost_after_every --> save costs while training after every get_cost_after_every epoch
# outputs:
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
    #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
    #                                               ...
    #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
    #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
    #                                               'B(N)': Nth layer bias of shape (1, 1)} 
    # - costs --> costs while training after every get_cost_after_every epoch
def train_on_multiple_images(linearized_images, layers, labels, activation='ReLU', epochs=1000, learning_rate=0.01, seed=None, print_loss_flag=False, print_after_epochs=100, get_cost_after_every=1):
    #################### START: initializations ####################
    
    parameters_dictionary = {}
    #for making reproducible results
    np.random.seed(seed)
    #will be returning this for making plots of costs while training.
    costs = []

    #looping  over len(layers) - 1 for weights initilizations, excluding the initial input layer as it doesn't have
    #   any weights
    for layer_index in range(len(layers) - 1):
        #Remember for weight initializations, we want number of nodes of current layer of which we are
        #   defining weights and also of the previous layer as both of these will be deciding the 
        #   shape of weight and bias matrix

        #current layer number will be layer_index + 1, since layer_index would start from 0 
        #Which means current_layer_num_of_nodes will be having num of nodes starting from 
        #   layer 1 to Nth layer (final output layer). We will be using current_layer_num_of_nodes
        #   for weights dimensions and also for naming weights and biases in parameters_dictionary
        current_layer_num_of_nodes = layers[layer_index + 1]
        #previous_layer_num_of_nodes will be having num of nodes starting from input layer 
        #   to (N-1)th layer (second last layer)
        previous_layer_num_of_nodes = layers[layer_index]
        #np.uniform will be producing weights matrix of given size with numbers [-1, 1) i.e including -1 but not 1
        #multiplying each weight with scalar to reduce its value further.
        parameters_dictionary['W' + str(layer_index + 1)] = np.random.uniform(low=-1, high=1, size=(previous_layer_num_of_nodes, current_layer_num_of_nodes)) * 1e-2
        parameters_dictionary['B' + str(layer_index + 1)] = np.random.uniform(low=-1, high=1, size=(current_layer_num_of_nodes, 1)) * 1e-2

    #################### END: initializations ####################

    #################### START: Training loop ####################

    #looping over how many training iterations do we want
    for epoch_index in range(epochs):
        #calculating network outputs for all samples with the weights defined in parameters_dictionary.
        #it ouputs cache which holds the Z's and A's (non-activated and activated) outputs for each node
        #   of each layer.
        cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, layers=layers, activation=activation)
        #calculate derivatives for each weight using cache produced earlier
        derivatives_dictionary = calculate_derivatives(network_inputs=linearized_images, labels=labels, activation=activation, parameters_dictionary=parameters_dictionary, cache=cache, layers=layers, threshold_flag=True, threshold=5)
        #update the weights
        parameters_dictionary = update_parameters(parameters_dictionary, derivatives_dictionary, layers, learning_rate)
        #appending the produced cost/loss, with the updated weights, for producing plot
        if np.mod(epoch_index, get_cost_after_every) == 0:
            costs.append(calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, layers, activation))
        #printing cost
        if np.mod(epoch_index, print_after_epochs) == 0 and print_loss_flag:
            print('epoch = ', epoch_index, '\nLoss = ', calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, layers, activation))

    #################### END: Training loop ####################

    return parameters_dictionary, costs

#%%Function to update parameters (weights and bias)
# input:
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
    #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
    #                                               ...
    #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
    #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
    #                                               'B(N)': Nth layer bias of shape (1, 1)} 
    # -derivatives_dictionary --> dictionary having {'dW1': derivative of Loss w.r.t W1 of shape(num of inputs, num of nodes in first layer), 
    #                                               'dB1': derivative of Loss w.r.t B1 of shape(num of nodes in first layer, 1),
    #                                               'dW2': derivative of Loss w.r.t W2 of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'dB2': derivative of Loss w.r.t B2 of shape(num of nodes in second layer, 1),
    #                                               ...
    #                                               'dW(N-1)': derivative of Loss w.r.t W(N-1) of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'dB(N-1)': derivative of Loss w.r.t B(N-1) of shape (num of nodes in (N-1)st layer, 1),
    #                                               'dW(N)': derivative of Loss w.r.t W(N) of shape(num of nodes in (N-1)st layer, 1),
    #                                               'dB(N)': derivative of Loss w.r.t B(N) of shape (1, 1)} 
    # -learning_rate --> learning rate for updation of parameters
# output:
    # -parameters_dictionary --> updated dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
            #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
            #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
            #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
            #                                               ...
            #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
            #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
            #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
            #                                               'B(N)': Nth layer bias of shape (1, 1)} 
def update_parameters(parameters_dictionary, derivatives_dictionary, layers, learning_rate):
    #we want to access all weights.
    #total number of weights are total number of layers - 1 because input layer doesn't
    #have any weights associated
    num_of_layers = len(layers) - 1
    for layer_index in range(num_of_layers):
        current_layer_num = layer_index + 1
        W = parameters_dictionary['W' + str(current_layer_num)]
        B = parameters_dictionary['B' + str(current_layer_num)]
        dW = derivatives_dictionary['dW' + str(current_layer_num)]
        dB = derivatives_dictionary['dB' + str(current_layer_num)]
        
        #update weights
        W = W - learning_rate * dW
        B = B - learning_rate * dB
        
        #saving the updated weights in the same parameters_dictionary
        parameters_dictionary['W' + str(current_layer_num)] = W
        parameters_dictionary['B' + str(current_layer_num)] = B
    
    return parameters_dictionary

#%% Function to calculate derivates for back propagation
#input:
    # -network_inputs --> network input of shape (height * width * num_of_channels, num of images)
    # -labels --> actuaL outputs, of shape (1, num of images)
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
    #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
    #                                               ...
    #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
    #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
    #                                               'B(N)': Nth layer bias of shape (1, 1)} 
    # -cache --> dictionary containing information from forward pass which would be required in calculating derivatives.
    #            {'Z1': non-activated nodes outputs of first layer of shape (num of nodes in first layer, num of samples),
    #             'A1': activated nodes outputs of first layer of shape (num of nodes in first layer, num of samples),
    #             'Z2': non-activated nodes outputs of second/output layer of shape(1, num of samples),
    #             'A2': activated nodes outputs of second/output layer of shape(1, num of samples),
    #              ...
    #             'Z(N-1)': non-activated nodes outputs of (N-1)st layer of shape (num of nodes in (N-1)st layer, num of samples),
    #             'A(N-1)': activated nodes outputs of (N-1)st layer of shape (num of nodes in (N-1)st layer, num of samples),
    #             'Z(N)': non-activated nodes outputs of Nth layer of shape(1, num of samples),
    #             'A(N)': activated nodes outputs of second layer of shape(1, num of samples)
    # -activation --> activation function being used in hidden layers
    # -threshold_flag --> whether to apply threshold on calculated derivative values or not, a bool
    # -threshold --> threshold to be applied on calculated derivative values
# output: 
    # -derivatives_dictionary --> dictionary having {'dW1': derivative of Loss w.r.t W1 of shape(num of inputs, num of nodes in first layer), 
    #                                               'dB1': derivative of Loss w.r.t B1 of shape(num of nodes in first layer, 1),
    #                                               'dW2': derivative of Loss w.r.t W2 of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'dB2': derivative of Loss w.r.t B2 of shape(num of nodes in second layer, 1),
    #                                               ...
    #                                               'dW(N-1)': derivative of Loss w.r.t W(N-1) of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'dB(N-1)': derivative of Loss w.r.t B(N-1) of shape (num of nodes in (N-1)st layer, 1),
    #                                               'dW(N)': derivative of Loss w.r.t W(N) of shape(num of nodes in (N-1)st layer, 1),
    #                                               'dB(N)': derivative of Loss w.r.t B(N) of shape (1, 1)} 
def calculate_derivatives(network_inputs, labels, parameters_dictionary, cache, layers, activation='ReLU', threshold_flag=False, threshold=1e-1):
    #see network_inputs definition
    #total num of samples in one epoch
    #will be used for taking out average derivatives
    num_of_images = np.shape(network_inputs)[1]
    #For layers = [100, 4, 6, 2, 1], num_of_hidden_layers will be 3
    num_of_hidden_layers = len(layers) - 2
    #For layers = [100, 4, 6, 2, 1], output_layer_num will be 4
    #Will be used for accessing output of network
    output_layer_num = num_of_hidden_layers + 1
    #Will be holding derivatives
    derivatives_dictionary = {}
    #network outputs for each sample
    network_outputs = cache['A' + str(output_layer_num)]
    #Derivative of Loss w.r.t final layer non-activated output Z(N)
    dZ_current_layer = network_outputs - labels

    #Need to start from weights of last layer.
    #[::-1] reverses the layer number count
    for layer_index in range(1, output_layer_num, 1)[::-1]:
        current_layer_num = layer_index + 1
        previous_layer_num = layer_index
        W_current_layer = parameters_dictionary['W' + str(current_layer_num)]
        A_previous_layer = cache['A' + str(previous_layer_num)]
        Z_previous_layer = cache['Z' + str(previous_layer_num)]
        
        dW_current_layer = (1/num_of_images) * np.matmul(A_previous_layer, dZ_current_layer.T)
        #summing all numbers in each row
        #If a = [[1, 2],
        #        [3, 4]]
        #then np.sum(a, axis=1) will output [[3],
        #                                    [7]]
        dB_current_layer = (1/num_of_images) * np.sum(dZ_current_layer, axis=1)
        dA_previous_layer = np.matmul(W_current_layer, dZ_current_layer)
        
        #dA/dZ where A and Z are of previous layer
        #will be used to calculate dL/dZ where Z is of previous layer, which will be
        #   used in next iteration
        #Copying the value Z_previous_layer values because otherwise its values would get
        #changed if dA_previous_layer_dZ_previous_layer is changed.
        dA_previous_layer_dZ_previous_layer = np.copy(Z_previous_layer)
        if activation == 'LeakyReLU':
            #dA/dZ would be 1 for all Z's > 0
            dA_previous_layer_dZ_previous_layer[dA_previous_layer_dZ_previous_layer >= 0] = 1
            #dA/dZ would be 0.1 for all Z's < 0
            dA_previous_layer_dZ_previous_layer[dA_previous_layer_dZ_previous_layer < 0] = 0.1
        elif activation == 'LeakyReLUReversed':
            #dA/dZ would be 0.1 for all Z's > 0
            dA_previous_layer_dZ_previous_layer[dA_previous_layer_dZ_previous_layer >= 0] = 0.1
            #dA/dZ would be 1 for all Z's < 0
            dA_previous_layer_dZ_previous_layer[dA_previous_layer_dZ_previous_layer < 0] = 1
        elif activation == 'Sigmoid':
            #dA/dZ = sigmoid(Z) * (1 - sigmoid(Z)) 
            dA_previous_layer_dZ_previous_layer = sigmoid(dA_previous_layer_dZ_previous_layer) * (1 - sigmoid(dA_previous_layer_dZ_previous_layer))
        elif activation == 'ReLU':
            #dA/dZ = 1  for all Z >= 0
            #Notice I have considered 0 too in this range.
            #Mathematically, ReLU's derivative is undefined at 0. But we have to give some
            #value to it for computations. Will not affect calculations to a great extent
            dA_previous_layer_dZ_previous_layer[dA_previous_layer_dZ_previous_layer >= 0] = 1
            #dA/dZ = 0  for all Z < 0
            dA_previous_layer_dZ_previous_layer[dA_previous_layer_dZ_previous_layer < 0] = 0
        if threshold_flag:
            dW_current_layer = np.clip(dW_current_layer, -threshold, threshold)
            dB_current_layer = np.clip(dB_current_layer, -threshold, threshold)
        
        derivatives_dictionary['dW' + str(current_layer_num)] = dW_current_layer
        #adding dimension to convert from (n, ) --> (n, 1)
        #always look out for these strange tuples. They give dimension errors when adding or multiplying with matrices
        #Making the tupe a numpy array of of 2 dimensions.
        derivatives_dictionary['dB' + str(current_layer_num)] = np.expand_dims(dB_current_layer, axis=1)
        
        #Now dZ_current_layer would used as the current layer dZ for the next iteration
        #just like how we add 1 in running index of a while loop to be used in next iteration
        dZ_current_layer = dA_previous_layer * dA_previous_layer_dZ_previous_layer
    
    #for first layer we are doing same calculation done in above loop
    #   because its values were not included in cache.
    #Note that A1, A2, ... act as input layers for the immediate next layer.
    #   There is no such thing as A0(our input) in cache which is required for calculated derivates
    #   of first layer
    #Here we treat input layer as A0 and do same calculations for it
    dW_current_layer = (1/num_of_images) * np.matmul(network_inputs, dZ_current_layer.T)
    dB_current_layer = (1/num_of_images) * np.sum(dZ_current_layer, axis=1)

    #adding an option for making sure our derivatives are not greater than in magnitude
    #   than some threshold amount.
    if threshold_flag:
        dW_current_layer = np.clip(dW_current_layer, -threshold, threshold)
        dB_current_layer = np.clip(dB_current_layer, -threshold, threshold)
    
    derivatives_dictionary['dW1'] = dW_current_layer
    #adding dimension to convert from (n, ) --> (n, 1)
    derivatives_dictionary['dB1'] = np.expand_dims(dB_current_layer, axis=1)
    
    return derivatives_dictionary

#%% calculate_derivatives test function
def test_calculate_derivatives():
    linearized_images = np.array([[2, 1], [5, 2], [1, 3], [6, 4]])
    layers = [4, 2, 2, 1]
    W1= np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    B1 = np.array([[1.0], [2.0]])
    W2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    B2 = np.array([[1.0], [2.0]])
    W3 = np.array([[1.0], [2.0]])
    B3 = np.array([[1.0]])
    parameters_dictionary = {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2, 'W3': W3, 'B3': B3}

    cache = calculate_network_output(linearized_images, parameters_dictionary, layers=layers, activation='LeakyReLUReversed')
    
    derivatives_dictionary = calculate_derivatives(network_inputs=linearized_images, labels=np.array([[0]]), parameters_dictionary=parameters_dictionary, cache=cache, layers=layers, activation='LeakyReLUReversed')
    a = 1
    
#%% Loss function implementation
# input:
    # linearized_images --> data on which to calculate loss, of shape (height*width*number of channels, number of images)
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
    #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
    #                                               ...
    #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
    #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
    #                                               'B(N)': Nth layer bias of shape (1, 1)} 
    # -labels --> images labels of shape (1, number of images)
    # -layers --> number of nodes in each layer, including first and last, a row vector e.g [250, 3, 2, 1] for 250 input features, 3 and 2 hidden layer units, and 1 output layer unit
    # -activation --> activation function being used in hidden layers
# output: loss, a float
def calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, layers, activation='ReLU'):
    num_of_images = np.shape(linearized_images)[1]
    cache = calculate_network_output(linearized_images, parameters_dictionary, layers, activation=activation)
    output_layer_num = len(layers) - 1
    network_outputs = cache['A' + str(output_layer_num)]
    #because zero was giving error when it got into log function since log(0) is undefined
    network_outputs[network_outputs == 0] = 1e-5
    network_outputs[network_outputs == 1] = 1-1e-5
    loss = -((labels * np.log(network_outputs)) + ((1 - labels) * np.log(1 - network_outputs)))
    loss = np.sum(loss)
    return loss/num_of_images

#%% Sigmoid function implementation
# x --> number to calculare sigmoid of
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#%% Function to linearize a 3-D image
# input:
    # images --> image of shape (num of images, height, width, num_of_channels) 
# output: 
    #linearized_images --> linearized images of shape (height * width * num_of_channels, num of images)
def linearize_images(images):
    # Extract dimensions of input image
    num_of_images = images.shape[0]
    image_height = images.shape[1]
    image_width = images.shape[2]
    image_channels = images.shape[3]
    
    linearized_images = images.reshape([num_of_images, image_height * image_width * image_channels])
    linearized_images = linearized_images.T
    
    return linearized_images
    
#%% Function for calculating output of Logistic Regression unit
# input:
    # -linearized_images --> images of shape (height * width * num_of_channels, num of images)
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
    #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
    #                                               ...
    #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
    #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
    #                                               'B(N)': Nth layer bias of shape (1, 1)} 
    # -layers --> number of nodes in each layer, including first and last, a row vector e.g [250, 3, 2, 1] for 250 input features, 3 and 2 hidden layer units, and 1 output layer unit
    # -activation --> activation function being used in hidden layers
# output:
    # -cache --> dictionary containing information from forward pass which would be required in calculating derivatives.
    #            {'Z1': non-activated nodes outputs of first layer of shape (num of nodes in first layer, num of samples),
    #             'A1': activated nodes outputs of first layer of shape (num of nodes in first layer, num of samples),
    #             'Z2': non-activated nodes outputs of second/output layer of shape(1, num of samples),
    #             'A2': activated nodes outputs of second/output layer of shape(1, num of samples),
    #              ...
    #             'Z(N-1)': non-activated nodes outputs of (N-1)st layer of shape (num of nodes in (N-1)st layer, num of samples),
    #             'A(N-1)': activated nodes outputs of (N-1)st layer of shape (num of nodes in (N-1)st layer, num of samples),
    #             'Z(N)': non-activated nodes outputs of Nth layer of shape(1, num of samples),
    #             'A(N)': activated nodes outputs of second layer of shape(1, num of samples)
            
def calculate_network_output(linearized_images, parameters_dictionary, layers, activation='ReLU'):

    cache = {}
    num_of_hidden_layers = len(layers) - 2
    A = linearized_images
    for layer_index in range(num_of_hidden_layers):
        layer_num = layer_index + 1
        W = parameters_dictionary['W' + str(layer_num)]
        B = parameters_dictionary['B' + str(layer_num)]
        
        Z = np.matmul(W.T, A) + B
        
        A = np.copy(Z)
        if activation == 'LeakyReLU':
            A[np.where(A < 0)] = A[np.where(A < 0)] * 0.1
        elif activation == 'LeakyReLUReversed':
            A[np.where(A > 0)] = A[np.where(A > 0)] * 0.1
        elif activation == 'ReLU':
            A[A < 0] = 0
        elif activation == 'Sigmoid':
            A = sigmoid(A)
            
        cache['Z' + str(layer_num)] = Z
        cache['A' + str(layer_num)] = A
        
    output_layer_number = num_of_hidden_layers + 1
    W = parameters_dictionary['W' + str(output_layer_number)]
    B = parameters_dictionary['B' + str(output_layer_number)]
    
    Z = np.matmul(W.T, A) + B
    A = np.copy(Z)
    A = sigmoid(A)
    
    cache['Z' + str(output_layer_number)] = Z
    cache['A' + str(output_layer_number)] = A
    
    return cache

#%% logistic_unit_output test function
    #tests logistic unit output for a 4 node input on one hidden layer network wiht 2 hidden units
def test_calculate_network_output():
    linearized_images = np.array([[1], [2], [3], [4]])
    layers = [4, 2, 2, 1]
    W1= np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    B1 = np.array([[1.0], [2.0]])
    W2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    B2 = np.array([[1.0], [2.0]])
    W3 = np.array([[1.0], [2.0]])
    B3 = np.array([[1.0]])
    parameters_dictionary = {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2, 'W3': W3, 'B3': B3}
#    linearized_images, parameters_dictionary, layers, activation='ReLU'
    cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, layers=layers, activation='LeakyReLUReversed')
    a = 1

#%%Function to test model on  multiple images
# input:
    # -linearized_images --> linearized images of shape (height * width * num_of_channels, num of images)
    # -labels --> image labels of shape (1, number of images)
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of nodes in first layer, 1),
    #                                               'W2': second layer weights of shape(num of nodes in first layer, num of nodes in second layer),
    #                                               'B2': second layer bias of shape (num of nodes in second layer, 1),
    #                                               ...
    #                                               'W(N-1)': (N-1)th layer weights of shape(num of nodes in (N-2)nd layer, num of nodes in (N-1)st layer)
    #                                               'B(N-1)': (N-1)th layer bias of shape (num of nodes in (N-1)st layer, 1),
    #                                               'W(N)': Nth layer weights of shape(num of nodes in (N-1)st layer, 1),
    #                                               'B(N)': Nth layer bias of shape (1, 1)} 
    # -layers --> number of nodes in each layer, including first and last, a row vector e.g [250, 3, 2, 1] for 250 input features, 3 and 2 hidden layer units, and 1 output layer unit
    # -activation --> activation function being used in first layer
    # -prediction_threshold --> threshold to apply on final network output for decision making
# output:
    # -logistic unit output, a number
def test_model_multiple_images(linearized_images, labels, parameters_dictionary, layers, activation='ReLU', prediction_threshold=0.5):
    num_of_images = linearized_images.shape[1]
    correct_predictions = 0
    output_layer_num = len(layers) - 1
    cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, layers=layers, activation=activation)
    network_outputs = cache['A' + str(output_layer_num)]
    network_outputs_before_threshold = np.copy(network_outputs)
    network_outputs[network_outputs >= prediction_threshold] = 1
    network_outputs[network_outputs < prediction_threshold] = 0
    correct_predictions = np.sum(network_outputs == labels)
            
    accuracy = correct_predictions/num_of_images
    return accuracy * 100, network_outputs, network_outputs_before_threshold


#%% Main Code

#Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

activation = 'LeakyReLU'
prediction_threshold = 0.7
learning_rate = 1e-3
epochs = 3000
seed = 1

linearized_train_set_x_orig = linearize_images(train_set_x_orig)
linearized_test_set_x_orig = linearize_images(test_set_x_orig)

num_input_layers = np.shape(linearized_train_set_x_orig)[0]
layers = [num_input_layers, 2, 2, 1]

parameters_dictionary, costs = train_on_multiple_images(linearized_images=linearized_train_set_x_orig, layers=layers, labels=train_set_y, activation=activation, epochs=epochs, learning_rate=learning_rate, seed=seed, print_loss_flag=True, print_after_epochs=50)

plt.plot(costs)
plt.show()

model_accuracy, network_outputs, network_outputs_before_threshold = test_model_multiple_images(linearized_train_set_x_orig, train_set_y, parameters_dictionary, layers, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(linearized_images=linearized_train_set_x_orig, parameters_dictionary=parameters_dictionary, labels=train_set_y, layers=layers, activation=activation)
print('Trained model accuracy on training set: ', model_accuracy)
print('Trained model loss on training set: ', model_loss)

model_accuracy, _ , _ = test_model_multiple_images(linearized_test_set_x_orig, test_set_y, parameters_dictionary, layers, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(linearized_images=linearized_test_set_x_orig, parameters_dictionary=parameters_dictionary, labels=test_set_y, layers=layers, activation=activation)
print('Trained model accuracy on test set: ', model_accuracy)
print('Trained model loss on test set: ', model_loss)

#%% Logistic unit output test
#test_calculate_network_output()

#%% calculate test_calculate_derivatives
#test_calculate_derivatives()