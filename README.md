# Deep Learning Project: Potato Disease Classification

![Potato Disease Classification](link-to-your-image.png)

This project aims to classify potato diseases using deep learning techniques. It includes data preprocessing, model creation, training, and evaluation. The goal is to create a model that can accurately identify diseases in potato plants based on images.

## Table of Contents
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Data Splitting](#data-splitting)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Model Compilation](#model-compilation)
- [Model Training](#model-training)
- [Plotting Training Curves](#plotting-training-curves)
- [Model Evaluation](#model-evaluation)
- [Prediction Function](#prediction-function)
- [Visualizing Predictions](#visualizing-predictions)
- [Model Saving](#model-saving)
- [Usage](#usage)
- [License](#license)

## Dataset
- The dataset used for this project contains images of potato plants with different diseases.
- You can find the dataset at [link-to-your-dataset](link-to-your-dataset).

## Data Preparation
- Images are loaded and organized into a dataset using TensorFlow's image_dataset_from_directory function.
- The dataset is shuffled and batched for training.

## Data Splitting
- The dataset is split into training, validation, and test sets using a custom function.
- The splits are typically 80% training, 10% validation, and 10% test.

## Data Preprocessing
- Data preprocessing includes resizing, rescaling, and data augmentation.
- Images are resized to a consistent size and rescaled to have pixel values in the range [0, 1].
- Data augmentation techniques like random flips and rotations are applied to increase the diversity of training data.

## Model Architecture
- The model architecture is a convolutional neural network (CNN).
- It consists of convolutional layers, max-pooling layers, and dense layers.
- The final layer has softmax activation for classification.

## Model Compilation
- The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric.

## Model Training
- The model is trained for a specified number of epochs.
- Training progress is monitored, and validation accuracy and loss are tracked.

## Plotting Training Curves
- Training and validation accuracy and loss curves are plotted to visualize model performance.

## Model Evaluation
- The trained model is evaluated on a batch of test data.
- Actual labels, predicted labels, and confidence scores are displayed for visual inspection.

## Prediction Function
- A prediction function is defined to make predictions on individual images.

## Visualizing Predictions
- Test images are displayed alongside their predicted classes and confidence scores.

## Model Saving
- The trained model is saved for future use or deployment.

## Usage
- Clone the repository: `git clone https://github.com/DINESHPANDIAN-J/Potato-Disease-Classification---CNN/edit/main/README.md`
- Run the Jupyter Notebook to train and evaluate the model.

