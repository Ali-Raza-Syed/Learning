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

def calculate_output(image):
    #%% Extract dimensions of input image
    print(image.shape)
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_channels= image.shape[2]
    
    linear_image = image.reshape([image_height * image_width * image_channels])
    linear_image = np.expand_dims(linear_image, axis=1)
    return linear_image

#%% check function
check = np.array([[[1, 2], [3, 4]], 
                  [[5, 6], [7, 8]],
                  [[9, 10], [11, 12]]])
print(calculate_output(check))
