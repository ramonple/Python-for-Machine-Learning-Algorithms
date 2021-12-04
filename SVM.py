import sklearn
from sklearn import datasets
from sklearn import svm


cancer = datasets.load_breast_cancer()
print(cancer.target_names)

X = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

classes = [ 'malignant','benign']

classifier = svm.SVC(kernel="linear")
#clf=svm.SVC(kernel = "ploy",degree = 2)  # as polynomial
classifier.fit(X_train, y_train)

y_pred = clf.predict(X_test)

import sklearn.metrics as metrics
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)
