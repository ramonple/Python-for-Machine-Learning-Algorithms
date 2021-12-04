# KNN for iris dataset
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv("iris.data",names=names)
# print(dataset.head())

# determine which are attributes and which is label
X = dataset.iloc[:,:-1].values # It means, select get all columns except the last column
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Feature Scaling
# Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train,y_train)

X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Training and Predictions
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
# The first step is to import the KNeighborsClassifier class from the sklearn.neighbors library.
# In the second line, this class is initialized with one parameter, i.e. n_neigbours.
# This is basically the value for the K. There is no ideal value for K and it is selected after testing and evaluation,
# however to start out, 5 seems to be the most commonly used value for KNN algorithm.

# The final step is to make predictions on our test data. To do so, execute the following script:
y_pred = classifier.predict(X_test)

# Evaluating the Algorithm
# For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics.
#  The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics.
#  Take a look at the following script:
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Comparing Error Rate with the K Value
# In the training and prediction section we said that there is no way to know beforehand which value of K that yields the best
# results in the first go. We randomly chose 5 as the K value and it just happen to result in 100% accuracy
# One way to help you find the best value of K is to plot the graph of K value and the corresponding error rate for the dataset.
# In this section, we will plot the mean error for the predicted values of test set for all the K values between 1 and 40.
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
# The above script executes a loop from 1 to 40.
# In each iteration the mean error for predicted values of test set is calculated and the result is appended to the error list.

# The next step is to plot the error values against K values. Execute the following script to create the plot:
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()