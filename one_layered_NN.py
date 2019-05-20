# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:59:30 2019

@author: Syed_Ali_Raza
"""
import numpy as np
from lr_utils import load_dataset
import warnings
warnings.filterwarnings("error")

#%% Function for training on multiple images
# inputs:
    # -linearized_images --> images to fit model to, of shape (height*width*number of channels, number of images)
    # -layers --> number of nodes in each layer, including first and last, a row vector e.g [250, 3, 2, 1] for 250 input features, 3 and 2 hidden layer units, and 1 output layer unit
    # -labels --> images labels of shape (1, number of images)
    # -epochs --> number of iterations for updating parameters, an integer
    # -learning_rate --> learning rate for updation of weights and bias
    # -seed --> seed value for random function. Default is None (for random), an integer
    # -print_loss_flag --> whether to print loss or not, bool
    # -print_after_epochs --> number of epochs after which to print the loss
# outputs:
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of inputs, 1),
    #                                               'W2': second/output layer weights of shape(num of nodes in first layer, 1),
    #                                               'B2': second/output layer bias of shape (1, 1)}   
def train_on_multiple_images(linearized_images, layers, labels, activation='ReLU', epochs=1000, learning_rate=0.01, seed=None, print_loss_flag=False, print_after_epochs=100):
#    for layer_index, current_layer_num_of_nodes in enumerate(layers[1:]):
    for layer_index in range(len(layers) - 1):
        current_layer_num_of_nodes = layers[layer_index + 1]
        previous_layer_num_of_nodes = layers[layer_index]
        parameters_dictionary['W' + str(layer_index + 1)] = np.random.uniform(low=-1, high=1, size=(previous_layer_num_of_nodes, current_layer_num_of_nodes)) * 1e-5
        parameters_dictionary['B' + str(layer_index + 1)] = np.random.uniform(low=-1, high=1, size=(current_layer_num_of_nodes, 1)) * 1e-5
    
    for epoch_index in range(epochs): 
        cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, layers=layers, activation=activation)
        derivatives_dictionary = calculate_derivatives(network_inputs=linearized_images, labels=labels, activation=activation, parameters_dictionary=parameters_dictionary, cache=cache, layers=layers, threshold_flag=False)
        parameters_dictionary = update_parameters(parameters_dictionary, derivatives_dictionary, layers, learning_rate)
        if np.mod(epoch_index, print_after_epochs) == 0 and print_loss_flag:
            print('epoch = ', epoch_index, '\nLoss = ', calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, layers, activation))
    return parameters_dictionary

#%%Function to update parameters (weights and bias)
# input:
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of inputs, 1),
    #                                               'W2': second/output layer weights of shape(num of nodes in first layer, 1),
    #                                               'B2': second/output layer bias of shape (1, 1)}
    # -derivatives_dictionary --> dictionary having {'dW1': derivative of W1 w.r.t loss, of shape (num of inputs, num of nodes in first layer), 
    #                                                'dB1': derivative of B1 w.r.t loss of shape (num of inputs, 1),
    #                                                'dW2': derivative of W2 w.r.t loss of shape(num of nodes in first layer, 1),
    #                                                'dB2': derivative of B2 w.r.t loss of shape (1, 1)}
    # -learning_rate --> learning rate for updation of parameters
# output:
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of inputs, 1),
    #                                               'W2': second/output layer weights of shape(num of nodes in first layer, 1),
    #                                               'B2': second/output layer bias of shape (1, 1)}
def update_parameters(parameters_dictionary, derivatives_dictionary, layers, learning_rate):
    
    num_of_layers = len(layers) - 1
    for layer_index in range(num_of_layers):
        current_layer_num = layer_index + 1
        W = parameters_dictionary['W' + str(current_layer_num)]
        B = parameters_dictionary['B' + str(current_layer_num)]
        dW = derivatives_dictionary['dW' + str(current_layer_num)]
        dB = derivatives_dictionary['dB' + str(current_layer_num)]
        
        W = W - learning_rate * dW
        B = B - learning_rate * dB
        
        parameters_dictionary['W' + str(current_layer_num)] = W
        parameters_dictionary['B' + str(current_layer_num)] = B
    
    return parameters_dictionary

#%% Function to calculate derivates for back propagation
#input:
    # -network_inputs --> network input of shape (height * width * num_of_channels, num of images)
    # -labels --> actuaL outputs, of shape (1, num of images)
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of inputs, 1),
    #                                               'W2': second/output layer weights of shape(num of nodes in first layer, 1),
    #                                               'B2': second/output layer bias of shape (1, 1)}
    # -cache --> dictionary containing information from forward pass which would be required in calculating derivatives.
    #            {'Z1': non-activated nodes outputs of first layer of shape (num of nodes in first layer, num of images),
    #             'A1': activated nodes outputs of first layer of shape (num of nodes in first layer, num of images),
    #             'Z2': non-activated nodes outputs of second/output layer of shape(1, num of images),
    #             'A2': activated nodes outputs of second/output layer of shape(1, num of images), which is also the network output}
    # -activation --> activation function being used in first layer
    # -threshold_flag --> whether to apply threshold on calculated derivative values or not, a bool
    # -threshold --> threshold to be applied on calculated derivative values
# output: 
    # -derivatives dictionary having {'dW1': derivative of W1 w.r.t loss, of shape (num of inputs, num of nodes in first layer), 
    #                                 'dB1': derivative of B1 w.r.t loss of shape (num of inputs, 1),
    #                                 'dW2': derivative of W2 w.r.t loss of shape(num of nodes in first layer, 1),
    #                                 'dB2': derivative of B2 w.r.t loss of shape (1, 1)}
def calculate_derivatives(network_inputs, labels, parameters_dictionary, cache, layers, activation='ReLU', threshold_flag=False, threshold=1e-1):
    num_of_images = np.shape(network_inputs)[1]
    num_of_hidden_layers = len(layers) - 2
    output_layer_num = num_of_hidden_layers + 1
    derivatives_dictionary = {}
    network_outputs = cache['A' + str(output_layer_num)]
    
    dZ = network_outputs - labels
    A = cache['A' + str(num_of_hidden_layers)]
    #excluding 1st layer
    for layer_index in range(1, output_layer_num, 1)[::-1]:
        current_layer_num = layer_index + 1
        previous_layer_num = layer_index
        W = parameters_dictionary['W' + str(current_layer_num)]
        A = cache['A' + str(previous_layer_num)]
        Z = cache['Z' + str(previous_layer_num)]
        
        dW = (1/num_of_images) * np.matmul(A, dZ.T)
        dB = (1/num_of_images) * np.sum(dZ, axis=1)
        dA = np.matmul(W, dZ)
        
        dA_dZ = np.copy(Z)
        if activation == 'LeakyReLU':
            dA_dZ[dA_dZ >= 0] = 1
            dA_dZ[dA_dZ < 0] = 0.1
        elif activation == 'LeakyReLUReversed':
            dA_dZ[dA_dZ >= 0] = 0.1
            dA_dZ[dA_dZ < 0] = 1
        elif activation == 'Sigmoid':
            dA_dZ = sigmoid(dA_dZ) * (1 - sigmoid(dA_dZ))
        elif activation == 'ReLU':
            dA_dZ[dA_dZ >= 0] = 1
            dA_dZ[dA_dZ < 0] = 0
            
        if threshold_flag:
            dW = np.clip(dW, -threshold, threshold)
            dB = np.clip(dB, -threshold, threshold)
        
        derivatives_dictionary['dW' + str(current_layer_num)] = dW
        #adding dimension to convert from (n, ) --> (n, 1)
        derivatives_dictionary['dB' + str(current_layer_num)] = np.expand_dims(dB, axis=1)
        
        dZ = dA * dA_dZ
    
    #for first layer
    dW = (1/num_of_images) * np.matmul(network_inputs, dZ.T)
    dB = (1/num_of_images) * np.sum(dZ, axis=1)

    if threshold_flag:
        dW = np.clip(dW, -threshold, threshold)
        dB = np.clip(dB, -threshold, threshold)
    
    derivatives_dictionary['dW1'] = dW
    #adding dimension to convert from (n, ) --> (n, 1)
    derivatives_dictionary['dB1'] = np.expand_dims(dB, axis=1)
    
    return derivatives_dictionary

#%% calculate_derivatives test function
def test_calculate_derivatives():
    linearized_images = np.array([[1], [2], [3], [4]])
    # 2 nodes in the layer
    W1= np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    B1 = np.array([[1], [2]])
    W2 = np.array([[2], [3]])
    B2 = np.array([[2]])
    parameters_dictionary = {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}

    cache = calculate_network_output(linearized_images, parameters_dictionary, activation='ReLU')
    
#    network_inputs, labels, parameters_dictionary, cache, activation='ReLU', threshold_flag=False, threshold=1e-1
    derivatives_dictionary = calculate_derivatives(network_inputs=linearized_images, labels=np.array([[0]]), parameters_dictionary=parameters_dictionary, cache=cache)
    a = 1
    
#%% Loss function implementation
# input:
    # network_output --> network output
    # true_output --> actual output
# output: loss, a float
def calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, layers, activation='ReLU'):
    num_of_images = np.shape(linearized_images)[1]
    cache = calculate_network_output(linearized_images, parameters_dictionary, layers, activation=activation)
    output_layer_num = len(layers) - 1
    network_outputs = cache['A' + str(output_layer_num)]
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
# output: linear image of shape (height * width * num_of_channels, num of images)
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
    #                                               'B1': first layer biases of shape (num of inputs, 1),
    #                                               'W2': second/output layer weights of shape(num of nodes in first layer, 1),
    #                                               'B2': second/output layer bias of shape (1, 1)}
    
# output:
    # -cache, dictionary containing information from forward pass which would be required in calculating derivatives.
    #       {'Z1': non-activated nodes outputs of first layer of shape (num of nodes in first layer, 1),
    #        'A1': activated nodes outputs of first layer of shape (num of nodes in first layer, 1),
    #        'Z2': non-activated nodes outputs of second/output layer of shape(1, 1),
    #        'A2': activated nodes outputs of second/output layer of shape(1, 1), which is also the network output}
            
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
    # 2 nodes in the layer
    W1= np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    B1 = np.array([[1], [2]])
    W2 = np.array([[2], [3]])
    B2 = np.array([[2]])
    parameters_dictionary = {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}
    cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, activation='ReLU')
    a = 1

#%%Function to test model on  multiple images
# input:
    # -images --> images of shape (num of images, image number, height, width, num_of_channels)
    # -labels --> image labels of shape (1, number of images)
    # -parameters_dictionary --> dictionary having {'W1': first layer weights of shape (num of inputs, num of nodes in first layer), 
    #                                               'B1': first layer biases of shape (num of inputs, 1),
    #                                               'W2': second/output layer weights of shape(num of nodes in first layer, 1),
    #                                               'B2': second/output layer bias of shape (1, 1)}
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
    network_outputs[network_outputs >= prediction_threshold] = 1
    network_outputs[network_outputs < prediction_threshold] = 0
    correct_predictions = np.sum(network_outputs == labels)
            
    accuracy = correct_predictions/num_of_images
    return accuracy * 100, network_outputs


#%% Main Code

#Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

activation = 'ReLU'
prediction_threshold = 0.7
learning_rate = 1e-2
epochs = 300
seed = 1

linearized_train_set_x_orig = linearize_images(train_set_x_orig)
linearized_test_set_x_orig = linearize_images(test_set_x_orig)

num_input_layers = np.shape(linearized_train_set_x_orig)[0]
layers = [num_input_layers, 4,  1]

parameters_dictionary = train_on_multiple_images(linearized_images=linearized_train_set_x_orig, layers=layers, labels=train_set_y, activation=activation, epochs=epochs, learning_rate=learning_rate, seed=seed, print_loss_flag=True, print_after_epochs=50)

model_accuracy, network_outputs = test_model_multiple_images(linearized_train_set_x_orig, train_set_y, parameters_dictionary, layers, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(linearized_images=linearized_train_set_x_orig, parameters_dictionary=parameters_dictionary, labels=train_set_y, layers=layers)
print('Trained model accuracy on training set: ', model_accuracy)
print('Trained model loss on training set: ', model_loss)

model_accuracy, _ = test_model_multiple_images(linearized_test_set_x_orig, test_set_y, parameters_dictionary, layers, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(linearized_images=linearized_test_set_x_orig, parameters_dictionary=parameters_dictionary, labels=test_set_y, layers=layers)
print('Trained model accuracy on test set: ', model_accuracy)
print('Trained model loss on test set: ', model_loss)

#%% Logistic unit output test
#test_logistic_unit_output()

#%% calculate test_calculate_derivatives
#test_calculate_derivatives()