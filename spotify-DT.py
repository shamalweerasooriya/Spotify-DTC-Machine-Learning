import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv(
    '/path/data/data.csv')

train, test = train_test_split(data, test_size=0.15)  # split %15 for testing

""" 
The goal of the tree is to ultimately split observations
into grouos of homogenous target values (1/0) giving us 
a set of paths to determine if this user liked or disliked
a particular song
"""
c = DecisionTreeClassifier(min_samples_split=100)

# features to be considered by the DTC
features = ['danceability', 'loudness', 'valence', 'energy',
            'instrumentalness', 'acousticness', 'key', 'speechiness', 'duration_ms', 'mode', 'liveness', ]

x_train = train[features]
y_train = train['target']

x_test = test[features]
y_test = test['target']

# Generating the decision tree
dt = c.fit(x_train, y_train)

# Predict target for the training set
y_pred = c.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy of the DTC: ", accuracy*100)
