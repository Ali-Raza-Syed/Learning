import torch
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

dataset = torch.tensor(linearize_images(train_set_x_orig), dtype=torch.float32)
labels = torch.tensor(train_set_y.T, dtype=torch.float32)

np.random.seed(seed)
weights = np.random.uniform(low=-1, high=1, size=(num_input_features, 1)) * 1e-5
bias = np.random.uniform(low=-1, high=1, size=(1, 1)) * 1e-5

W = torch.tensor(weights, dtype=torch.float32, requires_grad=True, device=torch.device("cpu"))
B = torch.tensor(bias, dtype=torch.float32, requires_grad=True, device=torch.device("cpu"))

for t in range(500):
    Z = dataset.mm(W).add(torch.transpose(B, 0, 1))
    y_pred = torch.nn.functional.sigmoid(Z)
    loss = torch.nn.functional.binary_cross_entropy(input=y_pred, target=labels)
    print(t, loss.item())
    loss.backward()
    with torch.no_grad():
        W -= learning_rate * W.grad
        B -= learning_rate * B.grad
        W.grad.zero_()
        B.grad.zero_()