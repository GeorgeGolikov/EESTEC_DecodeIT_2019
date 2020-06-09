import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# we import the tsv file
# parameters: delimiter is the way we instruct pandas that the columns are sepparated by TABS
# quoting = 3 ignores the "
dataset = pd.read_csv('decodeNew.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# we will take only the relevant words and get rid of irrelevant words like 'the'
# we will remove punctuation
# steamy takes root of the words: loved -> love
# in the end we want to have less words in the BAG
# re - library that will help us clean the text
import re
# we remove the words like this, and, etc
# if we have the word 'love' it will be enough to understand if the participant likes the event or not
# we will use another library
import nltk
# we import the tool, which is a list of irellevant words; this list it's called stopwords
# download is a function which will take this list
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# we create a corpus from all the words
# we called it corpus; this is a well-known termn, and represents a collection of text which is used in ML
corpus = []
for i in range(0, 100):
   feedback = re.sub('[^a-zA-Z]', ' ', dataset['Feedback'][i])
   feedback = feedback.lower()
   # split the words as elements
   feedback = feedback.split()
   # remove the irrelevat words and take the roots of the words
   ps = PorterStemmer()
   # word for word in feedback means that for different words in the feedback we include these words in the list
   # we add ifnot to add the words only if they are not part of the stopwords list
   feedback = [ps.stem(word) for word in feedback if not word in stopwords.words('english')]
   # we reverse the list into a string
   feedback = ' '.join(feedback)
   corpus.append(feedback)
   
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#*** be careful when using max_features **** this will take the word frequency - 1500 is a value to big

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)