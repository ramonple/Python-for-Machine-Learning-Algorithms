# decision tree
# https://stackabuse.com/decision-trees-in-python-with-scikit-learn/



# 1. Decision Tree for Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("bill_authentication.csv")
# print(dataset.head())

# prepare the data
X = dataset.drop('Class',axis=1)
y=dataset['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# training and making predictions
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier() # do not forget the ()
classifier.fit(X_train, y_train) # use the train x and y to get the fit model

# Now that our classifier has been trained, let's make predictions on the test data.
# To make predictions, the predict method of the DecisionTreeClassifier class is used
y_pred=classifier.predict(X_test) # use test set to train the model above


# Evaluating the Algorithm
# At this point we have trained our algorithm and made some predictions.
#  Now we'll see how accurate our algorithm is.
# For classification tasks some commonly used metrics are confusion matrix,
# precision, recall, and F1 score.
#  Lucky for us Scikit=-Learn's metrics library contains the classification_report
#  and confusion_matrix methods that can be used to calculate these metrics for us:

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix (y_test,y_pred))

# output:

#               precision    recall  f1-score   support
#
#            0       0.99      0.99      0.99       148
#            1       0.99      0.98      0.99       127
#
#     accuracy                           0.99       275
#    macro avg       0.99      0.99      0.99       275
# weighted avg       0.99      0.99      0.99       275
#
# [[142   2]
#  [  2 129]]

# From the confusion matrix, you can see that out of 275 test instances,
# our algorithm misclassified only 4. This is 98.5 % accuracy. Not too bad!


# 2. Decision Tree for Regression
dataset = pd.read_csv("petrol_consumption.csv")

# print(dataset.head())
# print(dataset.describe())
X = dataset.drop('Petrol_Consumption',axis=1)
y=dataset['Petrol_Consumption']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

# this time, differnt classifier
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

# Now let's compare some of our predicted values with the actual values and see how accurate we were:
answer=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(answer)

# Evaluating the Algorithm

# To evaluate performance of the regression algorithm, the commonly used metrics are
# mean absolute error,
# mean squared error,
# and root mean squared error.
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print(dataset.describe())