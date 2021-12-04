# KNN
from sklearn import preprocessing

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Loading Data
data=pd.read_csv("car.data")
print(data.head())

# Converting Data
# As you may have noticed much of our data is not numeric. In order to train the K-Nearest Neighbor Classifier we must
#  convert any string data into some kind of a number. Luckily for us sklearn has a method that can do this for us.

# We will start by creating a label encoder object and then use that to encode each column of our data into integers.
le=preprocessing.LabelEncoder()

# The method fit_transform() takes a list (each of our columns) and will return to us an array containing our new values.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# Now we need to recombine our data into a feature list and a label list. We can use the zip() function to makes things easier.
# The zip() function takes iterables (can be zero or more), aggregates them in a tuple, and returns it.
X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)

# Training a KNN Classifier
# from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)

model.fit(x_train,y_train)
acc=model.score(x_test,y_test)
print(acc)

# Testing Our Model
'''predicted = model.predict(x_test)
names =["unacc","acc","good","vgood"] #vgood very good

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])'''

# This will display the predicted class, our data and the actual class
# We create a names list so that we can convert our integer predictions into
# their string representation


# Looking at Neighbors
# The KNN model has a unique method that allows for us to see the neighbors of a given data point.
# We can use this information to plot our data and get a better idea of where our model may lack accuracy.
# We can use model.neighbors to do this.
#
# Note: the .neighbors method takes 2D as input, this means if we want to pass one data point we need surround it
# with [] so that it is in the right shape.
# Parameters: The parameters for .neighbors are as follows: data(2D array), # of neighbors(int), distance(True or False)
# Return: This will return to us an array with the index in our data of each neighbor.
# If distance=True then it will also return the distance to each neighbor from our data point.

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"] # refer to the data set, column 'class'

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # Now we will we see the neighbors of each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)