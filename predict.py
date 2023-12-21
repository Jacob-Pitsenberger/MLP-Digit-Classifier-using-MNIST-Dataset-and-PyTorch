"""
Module: predict.py

This module provides functions for making predictions using trained neural network models. It includes a prediction
function and a main function for demonstrating predictions with models trained with and without a validation set.

Functions:
- predict(model: Net, images: torch.Tensor, labels: torch.Tensor) -> None:
  Makes predictions using a trained neural network model and visualizes the results.
- main() -> None:
  Main function to demonstrate predictions with models trained with and without a validation set.
"""

import torch
from net import Net
import numpy as np
import matplotlib.pyplot as plt
from load_and_visualize_data import get_loaders_no_validation, get_batch


def predict(model: Net, images: torch.Tensor, labels: torch.Tensor) -> None:
    """Make predictions using a trained neural network model and visualize the results.

    Args:
    - model (Net): Trained neural network model.
    - images (torch.Tensor): Batch of images for prediction.
    - labels (torch.Tensor): True labels corresponding to the images.

    Returns:
    None
    """
    # get sample outputs
    output = model(images)

    # convert output probabilities to predicted class
    _, preds = torch.max(output, 1)

    # prep images for display
    images = images.numpy()

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))

    # Iterate over the first 20 indices in the batch for visualization
    for idx in np.arange(20):
        # Create a subplot for each image in a 2x10 grid
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])

        # Display the image in grayscale using imshow
        ax.imshow(np.squeeze(images[idx]), cmap='gray')

        # Set the title for the subplot with the predicted and true labels
        title_text = "{} ({})".format(str(preds[idx].item()), str(labels[idx].item()))

        # Set the title color based on prediction correctness (green if correct, red if incorrect)
        title_color = "green" if preds[idx] == labels[idx] else "red"

        # Set the title of the subplot with the formatted text and color
        ax.set_title(title_text, color=title_color)

    # Display the entire plot
    plt.show()


def main() -> None:
    """Main function to demonstrate predictions with models trained with and without a validation set.

    Returns:
    None
    """
    # Instantiate models and load pre-trained weights
    model_nv = Net()
    model_nv.load_state_dict(torch.load('output/model_no_validation.pt'))

    model_v = Net()
    model_v.load_state_dict(torch.load('output/model_with_validation.pt'))

    # get our test_loader, either method call to get this works (validation or no validation - same test set regardless)
    train_loader, test_loader = get_loaders_no_validation()

    # get a batch of images and their labels from the test data.
    images, labels = get_batch(test_loader)

    # Predict with the model that wasn't trained with a validation set.
    predict(model_nv, images, labels)

    # Predict with the model that was trained with a validation set.
    predict(model_v, images, labels)


if __name__ == "__main__":
    main()


