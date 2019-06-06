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

model = torch.nn.Sequential(
    torch.nn.Linear(num_input_features, 1),
    torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss()

learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(500):
    y_pred = model(dataset)

    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()