# Computer Vision with PyTorch

This project demonstrates building, training, and evaluating three deep learning models using PyTorch to classify images from the FashionMNIST dataset. It includes various functionalities, such as building Convolutional Neural Networks (CNNs), computing metrics, visualizing results, and saving/loading models.

## Requirements

- PyTorch
- torchvision
- mlxtend
- tqdm
- pandas
- matplotlib

## Getting Started

To get started, clone this repository and install the required packages. 

If running on Google Colab, you should not have to manually download any packages or datasets.

## Training and Evaluating Models

The main code includes functions to train, test, and evaluate models:

* train_step: Train the model for a single epoch.
* test_step: Evaluate the model on the test data.
* eval_model: Return a dictionary containing the results of the model predicting on the test data.
Training and evaluation loops are included for each of the three model architectures.

## Model Architectures

Three different models are implemented:

* Model 0: Simple Linear Neural Network.
* Model 1: Multilayer Feed-Forward Neural Network.
* Model 2: Convolutional Neural Network using the TinyVGG architecture.

## Visualization and Analysis

Various visualizations and analyses are included, such as:

* Visualizing predictions for random samples.
* Creating and plotting a confusion matrix.
* Comparing results across different models.

## Saving and Loading Models

Instructions for saving and loading trained models are provided.

* Save the trained model to a file.
* Load the model from the file.
* Evaluate the loaded model to ensure it is functioning as expected.
