"""
Module: net.py

This module defines the architecture of a neural network for digit classification. The network consists of three fully
connected layers with ReLU activation functions and dropout layers to prevent overfitting.

Class:
- Net: Neural network class with methods to initialize the architecture and define the forward pass.

Functions:
None

Attributes:
- fc1 (nn.Linear): First fully connected layer.
- fc2 (nn.Linear): Second fully connected layer.
- fc3 (nn.Linear): Third fully connected layer.
- dropout (nn.Dropout): Dropout layer to prevent overfitting.

Methods:
- __init__(): Initializes the neural network architecture.
- forward(x: Tensor) -> Tensor: Defines the forward pass of the neural network.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Net(nn.Module):
    """Neural Network Architecture for Digit Classification.

    This class defines the architecture of a neural network for digit classification.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
        dropout (nn.Dropout): Dropout layer to prevent overfitting.

    Methods:
        __init__(): Initializes the neural network architecture.
        forward(x: Tensor) -> Tensor: Defines the forward pass of the neural network.

    """

    def __init__(self):
        """Initialize the neural network architecture.

        Defines the layers and parameters of the neural network.

        """
        super(Net, self).__init__()

        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512

        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Define the forward pass of the neural network.

        Args:
            x (Tensor): Input tensor representing an image.

        Returns:
            Tensor: Output tensor representing the class scores.

        """
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x
