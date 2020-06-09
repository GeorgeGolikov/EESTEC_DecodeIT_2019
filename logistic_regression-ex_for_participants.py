# Logistic Regression - we will find out if Amazon customers are willing to buy a product or NOT

# RECAP: A class is the model of something we want to build. For example, if we make a house construction plan that gathers the instructions on how to build a house, then this construction plan is the class.
# RECAP: An object is an instance of the class. So if we take that same example of the house construction plan, then an object is simply a house. A house (the object) that was built by following the instructions of the construction plan (the class). And therefore there can be many objects of the same class, because we can build many houses from the construction plan.
# RECAP: A method is a tool we can use on the object to complete a specific action. So in this same example, a tool can be to open the main door of the house if a guest is coming. A method can also be seen as a function that is applied onto the object, takes some inputs (that were defined in the class) and returns some output.
# Happy coding :)


# Importing the libraries

# NUMPY - This library is a library that contains mathematical tools; basically this is the library that we need to include any types of mathematics in our code.
import numpy as np
# MATPLOTLIB - This library is a library that is going to help us plot very nice charts. It contains very intuitive and useful tools.
import matplotlib.pyplot as plt
# PANDAS - Pandas library is really the best library to import data sets and manage data sets.
import pandas as pd
# as pd is a shortcut




# Importing the dataset - we have over 400 observations; don't forget to set your workingspace in the same folder/path as your python script
dataset = pd.read_csv('ProbOfPurchase(ex1).csv')
# What is iloc? We are creating the matrix of features (consists only from 2 columns age and salary), as yesterday :) iloc and loc are operations for retrieving data from Pandas dataframes.
X = dataset.iloc[:, [2, 3]].values
# dependent variable vector which will be the prediction YES/NO for the purchase
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# We have to build 2 different sets: 1 for training the ML model and 1 for testing the ML model
# Performance on the test set should be almost the same as the one from training set; Why? Because the ML model will show us that it learned the correlations not memorise them
# 2 lines of code: import the library for this job, sklearn.
from sklearn.model_selection import train_test_split
# we will create X train = X training part of matrix of features, x test part of the matrix of features, y train= training part of y and y test = test part of y
# X_train and y_train are associated, they have the same indexes, same observations
# train_test_split needs the matrix X and the vector y - we add all the dataset; test_size = 0.5 means 50% data will go to training test. Good choice is 20% for the test size. From 10 observations, 2 will be tested, 8 will be used to traing the model
# random state is optional, we use this to have the same results in every team.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Remember Euclidian distance between 2 points
# We have to put the variables on the same scale!
# We will use standardisation which means -> for each observation and each feature, you withdraw the mean values of all the values of the features and divide it by the standard diviation
# Xstand = ( X - mean (x) ) / standard deviation (x)
# import a library for this, preprossing library and from this we will import class StandardScaler

from sklearn.preprocessing import StandardScaler

# create an object of this class sc 
# StandardScaler is a class
sc = StandardScaler()
# recompute X train, x test
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
# we will use the linear model library because the Logistic Regression is a Linear classifier, which means we are in 2D and our 2 categories of clients (buyers/not buyers) are separated by a straight line - see the grapf results later
from sklearn.linear_model import LogisticRegression
# the classifier is our logistic regression object
# check the LogisticRegression class for parameters - do we need any of them?
classifier = LogisticRegression()
# we are taking the classifier object and we are fitting it to the training set so that the classifier will learn the correlations between Xtrain and ytrain
classifier.fit(X_train, y_train)

# Predicting the Test set results
# 1 line, 1 variable y_pred is a vector of predections for each set of observations
# we take the classifier which was fitted and we will use the method from logistic regression class, which is PREDICT
# check the argument for this class. Which one is needed here?
y_pred = classifier.predict(X_test)


#----------- OPEN ALL THE VARIABLES and see the predictions --------------#



# We need to evaluate the performance of this model - correct vs incorrect predictions
# For this we will create the Confusion Matrix - correct predictions and incorrect predictions
# We will import a function from sklean.metrics library
# confusion_matrix = function DECODE IT TOP TIP: difference in coding, between a class and a function? The class name starts with capital letter
from sklearn.metrics import confusion_matrix
# check the parameters needed here?
cm = confusion_matrix(y_test, y_pred)
# the matrix was created, sum of the first diagonal represents the correct predictions, which are the incorrect ones?

# Visualising the Training set results
# In teams, describe the results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()