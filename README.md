# California Housing Median Value Prediction
This repository contains code for predicting the median house value in California using various machine learning models. The dataset used for this project is provided in the 'housing.csv' file.

## Overview
The goal of this project is to predict the median house value based on various features such as longitude, latitude, housing age, total rooms, total bedrooms, population, households, and median income. It is a regression problem since we are trying to predict a continuous value.

## Dependencies
- pandas
- numpy
- scikit-learn (for StandardScaler, LinearRegression, KNeighborsRegressor, RandomForestRegressor)
- matplotlib
- keras

## Dataset Preprocessing
The code starts by reading the dataset into a pandas DataFrame and then shuffling the data to randomize the order. It then drops the 'ocean_proximity' column and converts it into dummy variables to prepare the data for training.

Null values are dropped from the dataset, and the remaining data is split into training, validation, and testing sets.

## Data Scaling
The features in the dataset are scaled using the StandardScaler from scikit-learn. Scaling ensures that all features are on a similar scale, making the model's calculations more stable and accurate.

## Models
The following machine learning models are used for prediction:

1. Linear Regression: A simple regression algorithm that aims to minimize the difference between the predicted and actual values.

2. K-Nearest Neighbors Regression: A model that predicts the target value by looking at the 'k' nearest data points in the training data.

3. Random Forest Regression: A powerful ensemble method that creates a forest of decision trees and averages their predictions to make the final prediction.

4. Neural Network: A deep learning model that learns complex patterns and relationships within the data.

## Training, Validation, and Testing
The dataset is divided into three sets:

- Training: The model learns from this data during the training phase.
- Validation: The model's performance is checked using this separate set of data to tune hyperparameters and avoid overfitting.
- Testing: The model's real-world performance is evaluated on this unseen set of data.

## Usage
To run the code, make sure you have all the required dependencies installed. You can then execute the code to train the models and evaluate their performance on the validation set. The best model will be saved in the 'models' folder.

## Contributing
We welcome contributions to improve and optimize the models. Feel free to open an issue or submit a pull request with your suggestions or improvements.
