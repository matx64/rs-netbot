import numpy as np
from tqdm import tqdm # prints progress bar in terminal

import torch
import torch.optim as optim
import torch.nn as nn

def train(net):

    # Load Dataset
    dataset = np.load("dataset.npy", allow_pickle=True)

    # Dataset Images to Tensors & normalization
    X = torch.Tensor([i[0] for i in dataset]).view(-1, 50, 50)
    X = X/255.0

    # Loading labels
    Y = torch.Tensor([i[1] for i in dataset])

    # Adam Optimizer with 0.001 learning rate and MSE Loss function
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    BATCH_SIZE = 10
    EPOCHS = 20
    for epoch in range(EPOCHS):
        for i in tqdm(range(len(X), BATCH_SIZE)):
            batch_X = X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = Y[i:i+BATCH_SIZE]

            net.zero_grad() # reset gradients

            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()