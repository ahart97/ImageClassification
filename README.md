# Project Title: Webcam Image Classifier using CNNs

## Description

This repository contains a computer vision program that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, and then utilizes the trained model on images taken with a USB webcam. The program is a basic implementation and can be further expanded by using a more extensive dataset with higher quality images and additional classes, as well as a graphical user interface (GUI) to make the user interface more user-friendly.

## Installation
To run this program, you will need the following packages installed:

- Python 3
- NumPy
- TensorFlow
- OpenCV
- alive_progress
- matplotlib

You can install these packages using pip or another package manager.

## Usage
The repository contains the following files:

- CNN_model.py - this script is used to train and initialize the CNN on the CIFAR-10 dataset.
- main.py - this script is used to make predictions on images taken with a USB webcam.
- ComputerVision.py - this script is used to interface the usb webcam with a PC.
- signal_utils.py - useful signal utilies used in some of the scripts
- load_data.py - this script is used to load in the CIFAR-10 dataset

To run the program just use main.py, it will train and save a model if there is no model available, then run into the webcam loop.

## Contributing
If you would like to contribute to this repository, please fork the repository and make the necessary changes. Then, submit a pull request for review.
