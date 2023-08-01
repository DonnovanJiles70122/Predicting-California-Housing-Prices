import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from keras.models import load_model

'''
we are using this dataset to predict the median house value in california
this is a regression problem because we are trying to predict a continuous value
'''

pd.set_option('display.max_columns', None)
housing_df = pd.read_csv('housing.csv')

# randomizing the order of data in the DataFrame.
housing_df_shuffled = housing_df.sample(n=len(housing_df))

housing_df_updated = pd.concat([housing_df_shuffled.drop('ocean_proximity', axis=1),
                                pd.get_dummies(housing_df_shuffled['ocean_proximity'])], axis=1)

# print(housing_df_updated.head())
housing_df_updated = housing_df_updated[["longitude", "latitude", "housing_median_age",
                                         "total_rooms", "total_bedrooms", "population",
                                         "households", "median_income",
                                         "<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN", "median_house_value"]]

# drop null values in data set
housing_df_updated = housing_df_updated.dropna()

'''
Training: It's like a learning phase for the model. We give it lots of examples (data) 
and tell it what the correct answers are. The model learns from this data to make predictions.

Validation: After training, we need to check if the model is doing a good job. 
We use a separate set of data that the model hasn't seen before (validation data). 
The model makes predictions on this data, and we compare its predictions with the actual answers to see how accurate it is.

Testing: Finally, we want to see how well the model will perform in the real world on new, 
unseen data. So, we use another set of data (testing data) that the model has never encountered. 
We evaluate its performance and get an idea of how it will work when we use it for real-world predictions.
'''
train_data, testing_data, valuation_data = housing_df_updated[
    :18000], housing_df_updated[18000:19215], housing_df_updated[19215:]

# print(f"{len(train_data)} {len(testing_data)} {len(valuation_data)}")

# x_train will be a subset of the original dataset containing only the features (all columns in dataset except last),
# and y_train will be a separate array containing the corresponding target values (median_house_value).
# so we are preparing data for training this machine learning model where x_train is used as input features
# and y_train is the expected output for the model to make predictions
x_train, y_train = train_data.to_numpy()[:, :-1], train_data.to_numpy()[:, -1]

x_validation, y_validation = valuation_data.to_numpy(
)[:, :-1], valuation_data.to_numpy()[:, -1]

x_test, y_test = testing_data.to_numpy(
)[:, :-1], testing_data.to_numpy()[:, -1]

# print(f'{x_validation.shape} {y_validation.shape} {x_train.shape} {y_train.shape} {x_test.shape} {y_test.shape}')

'''
the Python StandardScaler is a preprocessing technique used to scale or normalize data in 
a way that makes it easier for machine learning models to understand and process.

The StandardScaler transforms the data so that each feature has a mean of 0 and a standard 
deviation of 1. This ensures all features are on a similar scale, making the model's calculations 
more stable and accurate.

By using StandardScaler, you prepare the data in a standardized format before feeding it into a 
machine learning model, helping the model perform better and improve the accuracy of its predictions.
'''
scaler = StandardScaler().fit(x_train[:, :8])


def preprocessor(X):
    A = np.copy(X)
    A[:, :8] = scaler.transform(A[:, :8])
    return A


X_train, X_validation, X_test = preprocessor(
    x_train), preprocessor(x_validation), preprocessor(x_test)
# print(f'{X_train_preprocessor}')

pd.DataFrame(X_train).hist()
# plt.show()

'''
mean squared error (MSE) is a way to measure how accurate a prediction model is 
by calculating the average of the squared differences between the predicted values 
and the actual values. It tells us how much the model's predictions deviate from the 
true values on average. A lower MSE indicates that the model's predictions are closer 
to the actual values and, therefore, more accurate.
'''

'''
overfitting in data science is when a machine learning model tries too hard to learn from 
the training data, becoming too specialized to that specific data. As a result, it performs 
well on the training data but poorly on new, unseen data, making it less useful for real-world 
predictions. It's like memorizing answers for a specific set of questions without understanding 
the underlying concepts, making it challenging to answer new questions correctly. To avoid 
overfitting, we need to ensure the model understands the general patterns in the data rather 
than just memorizing the examples it has seen.
'''

# The goal of linear regression is to minimize the difference between the predicted
# 'y' values and the actual 'y' values, allowing us to make accurate predictions or
# understand the trend between the two variables.

lm = LinearRegression().fit(X_train, y_train)
mse(lm.predict(X_train), y_train, squared=False)


'''
It predicts the value of a target variable by looking at the 'k' nearest data points in the 
training data. The predicted value is the average (or weighted average) of the target values 
of those 'k' nearest data points. The algorithm is useful when we want to predict a continuous 
numerical value based on the values of other features in the dataset. It is called 'KNeighbors' 
because it relies on the proximity of data points to make predictions.
'''
km = KNeighborsRegressor(n_neighbors=10).fit(X_train, y_train)
mse(km.predict(X_train), y_train, squared=False), mse(
    km.predict(X_validation), y_validation, squared=False)


'''
RandomForestRegressor is a machine learning algorithm used for regression tasks. 
It creates a "forest" of decision trees that work together to make predictions. 
Each tree in the forest makes its own prediction, and the final prediction is the 
average (or weighted average) of all the predictions from the individual trees.
'''


'''
RandomForestRegressor(max_depth=10): This line creates a random forest regression model 
with a maximum depth of 10. The model will learn from the X_train (input features) and 
y_train (target values) datasets during training.

.fit(X_train, y_train): This part of the code trains the model using the X_train and 
y_train datasets. It helps the model learn the patterns in the data so it can make predictions 
later.

rfr.predict(X_train): This line makes predictions using the trained model on the X_train dataset.

mse(...): This part calculates the Mean Squared Error (MSE) between the predicted values and 
the actual y_train values. The squared=False means it will give the Root Mean Squared Error (RMSE), 
which is a way to measure the accuracy of the model's predictions.

Similarly, the code calculates the RMSE for the predictions on the X_validation dataset and 
y_validation values.
'''
rfr = RandomForestRegressor(max_depth=10).fit(X_train, y_train)
mse(rfr.predict(X_train), y_train, squared=False), mse(
    rfr.predict(X_validation), y_validation, squared=False)

'''
neural networks are a type of computer algorithm inspired by the human brain's structure. 
They are used for solving complex problems and making predictions based on data.

Imagine a network of interconnected "neurons," where each neuron takes input, processes it, 
and produces an output. These interconnected neurons work together to learn patterns and 
relationships within the data.
'''

# initializes a neural network model called simple_nn with layers stacked one after another.
simple_nn = Sequential()

# adds an input layer to the model with 13 nodes. It defines the shape of the input data the model will work with.
simple_nn.add(InputLayer((13,)))

# This adds a hidden layer with 2 nodes to the model. The 'relu' activation function is used, which helps introduce
# non-linearity to the model and allows it to learn complex patterns in the data.
simple_nn.add(Dense(2, 'relu'))

# This adds an output layer with 1 node to the model. The 'linear' activation function is used,
# which means the output will be a continuous numerical value.
simple_nn.add(Dense(1, 'linear'))

# This line creates an optimizer called 'Adam', which helps adjust the model's parameters during training to minimize the error.
opt = Adam(learning_rate=.1)

# sets up a callback to save the best model during training. The model with the lowest validation
# loss will be saved in the 'models' folder with the name 'simple_nn'.
cp = ModelCheckpoint('models/simple_nn', save_best_only=True)

# This compiles the model by specifying the optimizer, loss function (Mean Squared Error),
# and metrics to measure during training (Root Mean Squared Error).
simple_nn.compile(optimizer=opt, loss='mse', metrics=[RootMeanSquaredError()])

# This line trains the model using the training data (X_train and y_train).
# The model's performance is evaluated using the validation data (X_validation and y_validation)
# during training. The model will be trained for 100 epochs (iterations) to learn from the data
# and minimize the error.
simple_nn.fit(x=X_train, y=y_train, validation_data=(
    X_validation, y_validation), callbacks=[cp], epochs=100)

simp_nn = load_model('models/simple_nn')
print(mse(simp_nn.predict(X_train), y_train, squared=False), mse(
    simp_nn.predict(X_validation), y_validation, squared=False))
