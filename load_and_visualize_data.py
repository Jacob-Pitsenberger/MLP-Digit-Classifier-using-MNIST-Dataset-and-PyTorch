"""
Author: Jacob Pitsenberger
Date: 12-21-23
Module: load_and_visualize_data.py

This module provides functions for downloading the MNIST dataset, loading data, and visualizing the images.

Functions:
- download_data(): Downloads the MNIST dataset.
- prepare_data_loader(train_data, sampler=None) -> torch.utils.data.DataLoader: Prepares a data loader for the given data and optional sampler.
- get_loaders_with_validation() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Prepares data loaders with a validation set.
- get_loaders_no_validation() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Prepares data loaders without a validation set.
- get_batch(loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]: Obtains one batch of training images from a data loader.
- view_images(images: torch.Tensor, labels: torch.Tensor): Plots a batch of images along with their labels.
- view_detailed_image(images: torch.Tensor): Displays a detailed view of a single image from the dataset.
- main(): Main function to demonstrate data loading and visualization.
"""

from typing import Tuple

import torch
import numpy as np
from six.moves import urllib
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

# Constants
BATCH_SIZE: int = 20
NUM_WORKERS: int = 0
VALIDATION_SIZE: float = 0.2


def download_data() -> None:
    """
    Downloads the MNIST dataset.
    Reference: https://github.com/pytorch/vision/issues/1938
    """
    # Create an opener with a custom user-agent to overcome download issues
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def prepare_data_loader(train_data, sampler=None) -> torch.utils.data.DataLoader:
    """
    Prepares a data loader for the given data and optional sampler.

    Args:
    - train_data: MNIST training data.
    - sampler: Optional sampler for the data loader.

    Returns:
    - DataLoader for the specified data.
    """
    # Use PyTorch DataLoader to load data in batches
    return torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)


def get_loaders_with_validation() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares data loaders with a validation set.

    Returns:
    - train_loader: DataLoader for the training set.
    - valid_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    """
    # Transform to convert images to PyTorch tensors
    transform = transforms.ToTensor()

    # Download and prepare MNIST training and test datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Shuffle training indices and split for validation set
    num_train = len(train_data)  # Get the total number of training samples
    indices = list(range(num_train))  # Create a list of indices corresponding to the training samples
    np.random.shuffle(indices)  # Shuffle the indices randomly

    # Calculate the split point for creating training and validation sets
    split = int(
        np.floor(VALIDATION_SIZE * num_train))  # Calculate the number of samples to include in the validation set

    # Split the indices into training and validation sets
    train_idx, valid_idx = indices[split:], indices[:split]  # Divide the indices into training and validation sets

    # Create data loaders for training, validation, and test sets
    train_loader = prepare_data_loader(train_data, SubsetRandomSampler(train_idx))
    valid_loader = prepare_data_loader(train_data, SubsetRandomSampler(valid_idx))
    test_loader = prepare_data_loader(test_data)

    return train_loader, valid_loader, test_loader


def get_loaders_no_validation() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares data loaders without a validation set.

    Returns:
    - train_loader: DataLoader for the training set.
    - test_loader: DataLoader for the test set.
    """
    # Transform to convert images to PyTorch tensors
    transform = transforms.ToTensor()

    # Download and prepare MNIST training and test datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Create data loaders for training and test sets
    train_loader = prepare_data_loader(train_data)
    test_loader = prepare_data_loader(test_data)

    return train_loader, test_loader


def get_batch(loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Obtains one batch of training images from a data loader.

    Args:
    - loader: DataLoader from which to obtain a batch.

    Returns:
    - images: Batch of images.
    - labels: Labels corresponding to the images.
    """
    # Retrieve the next batch of images and labels
    loader = loader
    images, labels = next(iter(loader))
    return images, labels


def view_images(images: torch.Tensor, labels: torch.Tensor) -> None:
    """
    Plots a batch of images along with their labels.

    Args:
    - images: Batch of images to be plotted.
    - labels: Labels corresponding to the images.
    """
    # Convert images to NumPy array for plotting
    images = images.numpy()

    # Plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))  # Create a new figure with a specified size for plotting

    # Iterate over the first 20 indices in the batch
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[],
                             yticks=[])  # Add a subplot for each image in a 2x10 grid
        ax.imshow(np.squeeze(images[idx]),
                  cmap='gray')  # Display the image, squeezing if necessary to remove single-dimensional entries

        # Print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))  # Set the title of the subplot with the correct label

    # Must add this to show when not working in Jupyter Notebooks.
    plt.show()


def view_detailed_image(images: torch.Tensor) -> None:
    """
    Displays a detailed view of a single image from the dataset.

    Args:
    - images: Batch of images.
    """
    # Convert images to NumPy array for detailed view
    images = images.numpy()

    # Extract a single image from the batch
    img = np.squeeze(images[1])

    # Create a detailed view plot for the image
    fig = plt.figure(figsize=(12, 12))  # Create a new figure with a specified size for detailed view
    ax = fig.add_subplot(111)  # Add a single subplot to the figure

    ax.imshow(img, cmap='gray')  # Display the image in grayscale

    width, height = img.shape  # Get the dimensions of the image
    thresh = img.max() / 2.5  # Set a threshold for determining text color based on pixel intensity

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0  # Round the pixel intensity to two decimal places
            # Annotate each pixel with its value
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',  # Set the horizontal alignment of text to the center of the annotation point
                        verticalalignment='center',  # Set the vertical alignment of text to the center of the annotation point
                        color='white' if img[x][y] < thresh else 'black')  # Set text color based on pixel intensity

    plt.show()  # Display the detailed view plot


def main() -> None:
    """
    Main function to demonstrate data loading and visualization.
    """
    # If the data isn't downloaded then do so.
    # download_data()

    # Prepare the data loaders without a validation set.
    train_loader, test_loader = get_loaders_no_validation()

    # Get a batch of images and their labels from the training data.
    images, labels = get_batch(train_loader)

    # View all the classes by getting some training images for them.
    view_images(images, labels)

    # View a single image from the training data in detail.
    view_detailed_image(images)


if __name__ == "__main__":
    main()
