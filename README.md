# Low-Light-Enhancement-GAN
DTU Computational Photography Project

Paper here: https://www.overleaf.com/read/mwfpmcfczqnd

## Aim
The aim of this project is to enhance poorly lit images into well lit images using a simple GAN architecture.

## Description

A GAN is a neural network composed of a generator and a discriminator, the generator tries to make candidate well-lit images from input poorly lit images. The discriminator tries to determine if what the generator outputs is similar to the ground-truth well-lit images from our dataset, we define a loss function that essentially optimizes the network to produce as high of an image relighting as possible.

## Dataset

We use this dataset: https://paperswithcode.com/dataset/lol

We split the "lol_dataset" into a folder called "high" and "low" for well-lit and poorly-lit images, respectively.

## lowlight metrics.py

This file has our most up to date code for dataloaders, attaining metrics, defining and training the network, and outputting progress pictures.

# inference.py

Does inference on selected input images. Will convert pickle.jpg into an enhanced output image.

# /models

Holds pickle (.pkl) files of machine learning models. Models are loaded by inference.py in order to use the trained model to perform generation from a given candidate poorly lit image.

