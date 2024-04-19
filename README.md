# Detecting Defects in Solar Panel Electroluminescence Images using Deep Learning

<p><strong><span style="font-size: larger;">Author:</span></strong> <a href="https://github.com/sebastianhoefler" style="font-size: larger;">Sebastian Hoefler</a></p>

[![Open Source Love](https://badges.frapsoft.com/os/v3/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/) 
This project is based on the Deep Learning challenge that was part of the Deep Learning course (SS 23) offered by the [Pattern Recognition Lab](https://lme.tf.fau.de/). The lecture videos for this course are available as a recording from Prof. Dr.-Ing. habil. Andreas Maier [here](https://www.youtube.com/watch?v=SCFToE1vM2U&list=PLpOGQvPCDQzvJEPFUQ3mJz72GJ95jyZTh). 

## Overview

Solar modules are composed of indivudal cells and are subject to degradation (transport, wind, hail, etc.). This can cause different defects such a cracks or inactive regions that can impact the functionality of the cells. A panel can have no or multiple defects (multi-label) and the defects are often not independent (although they are treated this way in this project).

The aim is to train a neural network capable of automatically determining which defects a solar module has. 


## Dataset

The dataset used in this project is a subset of the [ELPV dataset](https://github.com/zae-bayern/elpv-dataset). It contains 2000, 8-bit grayscale images of functional and defective solar cells. Three sample images from the dataset can be seen below: 

Left: Crack on a polycrystalline module; Middle: Inactive region; Right: Cracks and
inactive regions on a monocrystalline module

![alt text](<sample_images/sample_images.png>)

## Usage

`data.py` includes the data loader and performs augmentations for the training data. The transforms (augmentations) from torchvision were carefully chosen to fit the structure of the data. A sample of augmented images is displayed below.

![alt text](<sample_images/sample_aug_images.png>)


The code in `trainer.py`  defines a Trainer class, designed for training a deep learning model in PyTorch. It includes detailed methods for handling model training, validation/testing, logging, and checkpoint management. Here's a breakdown of its core functionalities:

* **Device configuration**: Sets GPU or Apple's Metal Performance Shaders (MPS) acceleration if CUDA is enabled.

* **Checkpoint Management**: Provides functionality to export the trained model to ONNX format for compatibility with different platforms or deployment environments.

* **Training and Validation/Testing Loops** `train_epoch` manages a single epoch of training, iterating over the training dataset, calculating loss, and updating the model parameters. `val_test` manages the validation/testing process, predicting outcomes on validation/test data, and computing the loss.

* **Performance Metrics**: Computes the F1 score for model performance evaluation, specifically designed to handle binary classification tasks such as defect detection in images.

* **TensorBoard Integration**: Uses SummaryWriter from the torch.utils.tensorboard module to log training and validation metrics for visualization in TensorBoard. This includes losses, learning rate, and F1 scores.

* **Training Execution (fit)**: Manages the overall training process across multiple epochs, handling early stopping, and logging results until completion or early termination based on the specified conditions.

The model should be run from `train.py`. Here, we can set the hyperparameters for training. Since there is a very big class imbalence, we oversample the minority class (undersample the majority class) to improve generalization. 

## Results

Here are the performance outcomes of the best model implementation:

| Dataset                  | Average F1-Score |
|--------------------------|------------------|
| Training Set             | 0.915            |
| **Hidden Challenge Set** | **0.909**        |

Before the challenge was taken offline, this repository ranked **second overall**.
