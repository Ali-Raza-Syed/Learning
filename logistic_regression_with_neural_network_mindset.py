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

#%% Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#%% Example of a picture
index = 13
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#%% Function for calculating output of Logistic Regression unit
# image --> image of shape (height, width, num_of_channels)
# weights --> weights to be multiplied of shape (num_of_weights, 1)
# bias --> bias to be added, a float value
def logistic_unit_output(image, weights, bias):
    
    #%% Extract dimensions of input image
    print(image.shape)
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_channels= image.shape[2]
    
    #%% Linearize image
    linear_image = image.reshape([image_height * image_width * image_channels])
    linear_image = np.expand_dims(linear_image, axis=1)

    #%% Weights transposed for multiplication
    # (number_of_weights, 1) --> (1, number_of_weights)
    weights_transposed = weights.T
    
    #%% Multiply weights with input and add bias
    # Note that bias is a float but gets broadcasted to a higher dimensional matrix for addition
    output = np.matmul(weights_transposed, linear_image) + bias
    
    #%% Squeezing output of shape (1, 1) to a single float number
    output = np.squeeze(output)
    
    return output

#%% check function
image = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]],
                  [[9, 10], [11, 12]]])

weights = np.array([[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]).T
bias = 2.5
logistic_output = logistic_unit_output(image, weights, bias)