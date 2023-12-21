# Project Overview

This project encompasses a series of modules designed to facilitate the creation, training, and prediction using a PyTorch MLP Neural Network for digit classification based on the MNIST dataset. The key modules include:

- **load_and_visualize_data.py**: Downloads the MNIST dataset, loads data, and provides visualization functions.
- **net.py**: Defines the architecture for a PyTorch MLP Neural Network.
- **train.py**: Offers functions for training the neural network with and without the inclusion of a validation dataset.
- **predict.py**: Provides functions for making predictions using trained models and visualizing the results.

Under the root directory "MLP Digit Classifier using MNIST Dataset and Pytorch," you'll find these modules, the 'data' directory (storing the MNIST dataset), and the 'output' directory (housing trained models).

# Purpose of the Project

This project serves as a practical exploration of PyTorch, undertaken during the Udacity course "Intro to Deep Learning with PyTorch." The motivation was to reinforce theoretical knowledge gained during prior studies in computer science and research papers focused on neural networks, particularly multi-layer perceptrons (MLPs), computer vision, and robotics.

While my earlier studies delved into the mathematical underpinnings of MLPs—covering matrix multiplication, summations, series, matrix normalization, backpropagation, gradients, gradient descent, and stochastic gradient descent, to name a few—I recognized a gap in practical machine learning experience. Despite holding an engineering science degree, I discovered that employers often sought explicit evidence of machine learning proficiency, prompting my enrollment in the Udacity course and the creation of projects like this one to showcase my practical skills.
To bridge this gap, I embarked on the Udacity course and created projects like this one, contributing to my public portfolio to showcase my practical skills.

# Usage

To utilize this project, you can either train your own models from scratch or use pre-trained models available in the 'output' directory.

## Starting From Scratch

### Prep Data

If you're not cloning the entire repository (with 'data' and 'output' directories), run the `download_data()` function in `load_and_visualize_data.py`. Create a 'output' directory under the root.

### Visualize MNIST Data

Ensure the `download_data()` function in `load_and_visualize_data.py` is commented out. Run the module as the main loop to execute the main function, creating data loaders, fetching a batch, and visualizing the data using `view_images(images, labels)` and `view_detailed_image(images)`.

### Train Model(s)

With downloaded data and proper directories, run the `train_network_no_validation` or `train_network_with_validation` functions in `train.py` to train the model (defined in `net.py`). The trained model's weights and information will be saved under the 'output' directory.

### Make Predictions

After training, run `predict.py`, ensuring the model dict specified in the main function is loaded from your trained model (.pt) file path.

## Using Pre-trained Models

If using the provided pre-trained models in the 'output' directory, run `predict.py` as the main loop to make predictions using both models. You can also visualize MNIST data using `load_and_visualize_data.py` as described in the 'Starting From Scratch' section.

Feel free to explore and adapt the project to your needs, and I hope you find it informative and enjoyable!

# Author
Jacob Pitsenberger - 2023

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

