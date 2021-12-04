import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


# linear Regression
data = pd.read_csv("student-mat.csv",sep=";") # sep argument will be used as separator or delimiter.
                                             # If sep argument is not specified then default engine for parsing ( C Engine) will be used which uses ‘,’ as delimiter
data = data[ ["G1","G2","G3","studytime", "failures","absences"] ]
predict = "G3"

X = np.array(data.drop( [predict], axis=1)) # select attributes. axis 0 for index, 1 for column
y=np.array(data[predict]) # select the dependent variable
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

best = 0
for _ in range(30): # select the best from 30 runs#
#must be a space before x_train
   x_train, x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.2) # split training and test dataset
                  # focus on the squence. first x and then y, first train and then test
linear = linear_model.LinearRegression()
linear.fit(x_train,y_train) # use the t_train and y_train to find the regression
acc = linear.score(x_test,y_test) # accuracy
print(acc)

if acc > best:
    best =acc  # also save the currently best one
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear,f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Coefficients: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
# Seeing a score value is cool but I'd like to see how well our algorithm works on specific students.
#  To do this we are going to print out all of our test data.
#  Beside this data we will print the actual final grade and our models predicted grade.
for  x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p1 = "G1"
style.use("ggplot")
pyplot.scatter(data[p1],data["G3"]) # G3 is the predictor
pyplot.xlabel(p1)
pyplot.ylabel ("final grage")
pyplot.show()

p2 = "failures"
style.use("ggplot")
pyplot.scatter(data[p2],data["G3"]) # G3 is the predictor
pyplot.xlabel(p2)
pyplot.ylabel ("final grage")
pyplot.show()