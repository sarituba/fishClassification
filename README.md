# Fish Species Classification Using Neural Networks

This project involves classifying different fish species based on various physical characteristics using an Artificial Neural Network (ANN). The primary goal is to build an accurate model that can differentiate between seven different fish species using the given dataset.
In this project, we will build a neural network to classify seafoods. We'll use the “A Large Scale Fish Dataset” available on Kaggle for this. The dataset contains 9 different seafood types of that can be classified into 9 classes according to their types.
## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Dataset

I'll use the “A Large Scale Fish Dataset” available on Kaggle for this. The dataset contains 9 different seafood types of that can be classified into 9 classes according to their types.

The dataset is publicly available on Kaggle: kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset.

## Model Architecture

In this project, we used an Artificial Neural Network (ANN) to classify fish species. The model architecture consists of:

- **Input Layer**: Flatten. This layer is used to convert the input images, which are in 3D shape (128x128 pixels with 3 color channels for RGB), into a 1D vector.
This is necessary because the Dense layers that follow expect 1D input.
- **Hidden Layers**: Two fully connected layers with ReLU activation functions and 1 Dropout function
- **Output Layer**: A softmax activation function to predict one of the 9 fish species.

### Model Details:
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

The data is split into training and test sets, and the model is trained to minimize the loss function, with the objective of maximizing classification accuracy.

