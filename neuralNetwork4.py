# Artificial Neural Network
# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras
# SECTION 1 - Data Preprocessing
# -----------------------------------------------------------------
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('dataset4.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
# Encoding categorical data
# take care of country variables, convert France, Spain, Germany in numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
# take care of gender variables, convert male,female in numbers
# we will have dummy variables
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# remove 1 dummy variable, we only need 2
X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling - needed ofc
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# SECTION 2 - BUILD the ANN
# -----------------------------------------------------------------
# Importing the Keras libraries and packages
import keras
# sequential module - initialize the ANN
from keras.models import Sequential
# dense module - build the layers of our ANN
from keras.layers import Dense
# Initialising the ANN - as a sequence of layers
# we will have a sequence of layers
# don't need to define the arguments, the layers will be added one by one
classifier = Sequential()
# Adding the input layer and the first hidden layer
# this ANN will be a classifier like the one from the classification exercise
# step 1 - dense function will take care of step 1
# step 2 - input_dim = 11 - these are the independent variables
# step 3 - choose an activation function - rectifer for hidden layers + sigmund functions for output layer
# units = 6 no of nodes - (no of independent variables + no of dependent variables)/2
# which means (11+1) / 2 = 6
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output hidden layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN, using the compile method
# adam? a type of  stochastic gradient descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# SECTION 3 - Making the predictions and evaluating the model
# -----------------------------------------------------------------
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)