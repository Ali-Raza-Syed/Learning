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
    # -num_of_nodes --> number of nodes to be made in a single layer
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
def train_on_multiple_images(linearized_images, num_of_nodes, labels, activation='ReLU', epochs=1000, learning_rate=0.01, seed=None, print_loss_flag=False, print_after_epochs=100):
    num_of_weights = np.shape(linearized_images)[0]
    np.random.seed(seed)
    W1 = np.random.uniform(low=-1, high=1, size=(num_of_weights, num_of_nodes)) * 1e-5  
    B1 = np.random.uniform(low=-1, high=1, size=(num_of_nodes, 1)) * 1e-5
    W2 = np.random.uniform(low=-1, high=1, size=(num_of_nodes, 1)) * 1e-5  
    B2 = np.random.uniform(low=-1, high=1, size=(1, 1)) * 1e-5
    parameters_dictionary = {'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}
    
    
    for epoch_index in range(epochs): 
        cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, activation=activation)
        derivatives_dictionary = calculate_derivatives(network_inputs=linearized_images, labels=labels, activation=activation, parameters_dictionary=parameters_dictionary, cache=cache, threshold_flag=False)
        
        parameters_dictionary = update_parameters(parameters_dictionary, derivatives_dictionary, learning_rate)
        if np.mod(epoch_index, print_after_epochs) == 0 and print_loss_flag:
            print('epoch = ', epoch_index, '\nLoss = ', calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, activation))
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
def update_parameters(parameters_dictionary, derivatives_dictionary, learning_rate):
    W1 = parameters_dictionary['W1']
    B1 = parameters_dictionary['B1']
    W2 = parameters_dictionary['W2']
    B2 = parameters_dictionary['B2']
    dW1 = derivatives_dictionary['dW1']
    dB1 = derivatives_dictionary['dB1']
    dW2 = derivatives_dictionary['dW2']
    dB2 = derivatives_dictionary['dB2']
    
    #updating parameters
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    
    parameters_dictionary['W1'] = W1
    parameters_dictionary['B1'] = B1
    parameters_dictionary['W2'] = W2
    parameters_dictionary['B2'] = B2
    
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
def calculate_derivatives(network_inputs, labels, parameters_dictionary, cache, activation='ReLU', threshold_flag=False, threshold=1e-1):
    
    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']
    
    W1 = parameters_dictionary['W1']
    B1 = parameters_dictionary['B1']
    W2 = parameters_dictionary['W2']
    B2 = parameters_dictionary['B2']
    
    network_outputs = cache['A2']
    
    num_of_images = np.shape(network_inputs)[1]
    
    # derivative of Loss w.r.t z, a number
    dZ2 = network_outputs - labels

    dW2 = (1/num_of_images) * np.matmul(A1, dZ2.T)
    dB2 = (1/num_of_images) * np.sum(dZ2, axis=1)
    
    dA1 = np.matmul(W2, dZ2)
    
    #derivative of activation function w.r.t z
    #shape will be (number of nodes in layer, 1)
#    activation_derivative = np.copy(z)
    d_A1_d_Z1 = np.copy(Z1)
    if activation == 'LeakyReLU':
        d_A1_d_Z1[d_A1_d_Z1 >= 0] = 1
        d_A1_d_Z1[d_A1_d_Z1 < 0] = 0.1
    elif activation == 'LeakyReLUReversed':
        d_A1_d_Z1[d_A1_d_Z1 >= 0] = 0.1
        d_A1_d_Z1[d_A1_d_Z1 < 0] = 1
    elif activation == 'Sigmoid':
        d_A1_d_Z1 = sigmoid(d_A1_d_Z1) * (1 - sigmoid(d_A1_d_Z1))
    elif activation == 'ReLU':
        d_A1_d_Z1[d_A1_d_Z1 >= 0] = 1
        d_A1_d_Z1[d_A1_d_Z1 < 0] = 0
    
    dZ1 = dA1 * d_A1_d_Z1
    
    dW1 = (1/num_of_images) * np.matmul(network_inputs, dZ1.T)
    dB1 = (1/num_of_images) * np.sum(dZ1, axis=1)
    dB1 = np.expand_dims(dB1, axis=1)
    if threshold_flag:
        dZ1 = np.clip(dZ1, -threshold, threshold)
        dB1 = np.clip(dB1, -threshold, threshold)
        dZ2 = np.clip(dZ2, -threshold, threshold)
        dB2 = np.clip(dB2, -threshold, threshold)
    
    derivatives_dictionary = {'dW1': dW1, 'dB1': dB1, 'dW2': dW2, 'dB2': dB2}
    
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
def calculate_loss_multiple_images(linearized_images, parameters_dictionary, labels, activation='ReLU'):
    num_of_images = np.shape(linearized_images)[1]
    cache = calculate_network_output(linearized_images, parameters_dictionary, activation=activation)
    network_outputs = cache['A2']
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
            
def calculate_network_output(linearized_images, parameters_dictionary, activation='ReLU'):
    W1 = parameters_dictionary['W1']
    B1 = parameters_dictionary['B1']
    W2 = parameters_dictionary['W2']
    B2 = parameters_dictionary['B2']

    # Weights transposed for multiplication
    # (number of nodes/pixels in image, number of nodes in the layer) --> (number of nodes in the layer, number of nodes/pixels in image)
#    weights_transposed = weights.T
    
    # Multiply weights with input and add bias
    # Note that bias is a vector but gets broadcasted to a higher dimensional matrix for addition
    Z1 = np.matmul(W1.T, linearized_images) + B1
    
#    parameters_dictionary['z'] = nodes_outputs
    
    #For introducing non-linearity. If hadn't done so, all the nodes sum would be same as using a single node
    A1 = np.copy(Z1)
#    prediction_before_activation = 0
    if activation == 'LeakyReLU':
        A1[np.where(A1 < 0)] = A1[np.where(A1 < 0)] * 0.1
    elif activation == 'LeakyReLUReversed':
        A1[np.where(A1 > 0)] = A1[np.where(A1 > 0)] * 0.1
    elif activation == 'ReLU':
        A1[A1 < 0] = 0
    elif activation == 'Sigmoid':
        A1 = sigmoid(A1)    
        
    Z2 = np.matmul(W2.T, A1) + B2
        
#    prediction_before_activation = np.sum(A1)
    # Take sigmoid of output
    A2 = sigmoid(Z2)
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    
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
def test_model_multiple_images(linearized_images, labels, parameters_dictionary, activation='ReLU', prediction_threshold=0.5):
    num_of_images = linearized_images.shape[1]
    correct_predictions = 0
    
    cache = calculate_network_output(linearized_images=linearized_images, parameters_dictionary=parameters_dictionary, activation=activation)
    network_outputs = cache['A2']
    network_outputs[network_outputs >= prediction_threshold] = 1
    network_outputs[network_outputs < prediction_threshold] = 0
    correct_predictions = np.sum(network_outputs == labels)
            
    accuracy = correct_predictions/num_of_images
    return accuracy * 100, network_outputs


#%% Main Code

#Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

activation = 'LeakyReLU'
prediction_threshold = 0.7
learning_rate = 0.5e-3
epochs = 700
seed = 1
num_of_nodes = 100

linearized_train_set_x_orig = linearize_images(train_set_x_orig)
linearized_test_set_x_orig = linearize_images(test_set_x_orig)

parameters_dictionary = train_on_multiple_images(linearized_images=linearized_train_set_x_orig, num_of_nodes=num_of_nodes, labels=train_set_y, activation=activation, epochs=epochs, learning_rate=learning_rate, seed=seed, print_loss_flag=True, print_after_epochs=50)

model_accuracy, network_outputs = test_model_multiple_images(linearized_train_set_x_orig, train_set_y, parameters_dictionary, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(linearized_images=linearized_train_set_x_orig, parameters_dictionary=parameters_dictionary, labels=train_set_y)
print('Trained model accuracy on training set: ', model_accuracy)
print('Trained model loss on training set: ', model_loss)

model_accuracy, _ = test_model_multiple_images(linearized_test_set_x_orig, test_set_y, parameters_dictionary, activation=activation, prediction_threshold=prediction_threshold)
model_loss = calculate_loss_multiple_images(linearized_images=linearized_test_set_x_orig, parameters_dictionary=parameters_dictionary, labels=test_set_y)
print('Trained model accuracy on test set: ', model_accuracy)
print('Trained model loss on test set: ', model_loss)

#%% Logistic unit output test
#test_logistic_unit_output()

#%% calculate test_calculate_derivatives
#test_calculate_derivatives()