"""
Module: train.py

This module provides functions for training a neural network using PyTorch. It includes functions for training with and
without a validation set.

Functions:
- train_network_with_validation(net_info, train_loader, valid_loader, n_epochs): Trains a neural network with a
  validation set.
- train_network_no_validation(net_info, train_loader, n_epochs): Trains a neural network without a validation set.
- main(): Main function to demonstrate training with and without a validation set.

"""

from load_and_visualize_data import get_loaders_with_validation, get_loaders_no_validation
from net import Net
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple


def train_network_with_validation(net_info: List, train_loader: DataLoader, valid_loader: DataLoader,
                                  n_epochs: int) -> None:
    """
    Trains a neural network with a validation set.

    Args:
    - net_info (List): List containing the neural network model, loss criterion, and optimizer.
    - train_loader (DataLoader): DataLoader for the training set.
    - valid_loader (DataLoader): DataLoader for the validation set.
    - n_epochs (int): Number of training epochs.

    Returns:
    None
    """
    print("TRAINING NETWORK WITH VALIDATION SET")
    model, criterion, optimizer = net_info

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss
            valid_loss += loss.item()

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        # Print training and validation loss for the current epoch
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,  # Epoch number (starting from 1)
            train_loss,  # Average training loss for the epoch
            valid_loss  # Average validation loss for the epoch
        ))

        # Save the model if the validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,  # Previous minimum validation loss
                valid_loss  # Current validation loss
            ))
            torch.save(model.state_dict(), 'output/model_with_validation.pt')  # Save the model's state dictionary
            valid_loss_min = valid_loss  # Update the minimum validation loss


def train_network_no_validation(net_info: List, train_loader: DataLoader, n_epochs: int) -> None:
    """
    Trains a neural network without a validation set.

    Args:
    - net_info (List): List containing the neural network model, loss criterion, and optimizer.
    - train_loader (DataLoader): DataLoader for the training set.
    - n_epochs (int): Number of training epochs.

    Returns:
    None
    """
    print("TRAINING NETWORK WITHOUT VALIDATION SET")
    model, criterion, optimizer = net_info

    model.train()  # prep model for training

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)

        # Print training statistics for the current epoch
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch + 1,  # Print the current epoch number (1-indexed)
            train_loss  # Print the training loss for the current epoch
        ))

        # Save the model's state dictionary after each epoch
        # This creates a snapshot of the model's parameters for future use or further training
        torch.save(model.state_dict(), 'output/model_no_validation.pt')


def main() -> None:
    """Main function for training and evaluating a neural network for digit classification.

    This function orchestrates the training process, including data loading, model initialization,
    and training loop execution. It uses the MNIST dataset, a neural network architecture defined in the 'Net' class,
    categorical cross-entropy loss, and stochastic gradient descent as the optimizer.

    The function trains two versions of the model: one without a validation set and another with a validation set.
    The trained models are saved after training.

    Returns:
        None
    """

    train_loader_v, valid_loader_v, test_loader_v = get_loaders_with_validation()

    train_loader_nv, test_loader_nv = get_loaders_no_validation()

    # initialize the NN
    model = Net()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    net_info = [model, criterion, optimizer]

    # number of epochs to train the model
    n_epochs = 20

    # Train a model without the validation set included.
    # train_network_no_validation(net_info, train_loader_nv, n_epochs)

    # Train a model with the validation set included.
    train_network_with_validation(net_info, train_loader_v, valid_loader_v, n_epochs)


if __name__ == "__main__":
    main()
