from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_frame = pd.read_csv("Banana.csv")
x = data_frame.iloc[:, 0:2].values
y = data_frame.iloc[:, 2].values


train_x, aux_x, train_y, aux_y = train_test_split(x, y, test_size=0.5)
validation_x, test_x, validation_y, test_y = train_test_split(aux_x, aux_y, test_size=0.5)

def update_data():
    global train_x
    global train_y
    global validation_x
    global test_x
    global validation_y
    global test_y

    train_x, aux_x, train_y, aux_y = train_test_split(x, y, test_size=0.5)
    validation_x, test_x, validation_y, test_y = train_test_split(aux_x, aux_y, test_size=0.5)

def knn():
    nbrs = KNeighborsClassifier(n_neighbors=11, algorithm='auto')
    nbrs.fit(train_x, train_y)
    return accuracy(test_y, nbrs.predict(test_x))


def dst():
    decision = DecisionTreeClassifier()
    decision = decision.fit(train_x, train_y)
    return accuracy(test_y, decision.predict(test_x))


def nb():
    bn = BernoulliNB()
    bn.fit(train_x, train_y)
    return accuracy(test_y, bn.predict(test_x))


def svm():
    svm = SVC()
    svm.fit(train_x, train_y)
    return accuracy(test_y, svm.predict(test_x))


def mlp():
    mlp = MLPClassifier(hidden_layer_sizes=8, activation='logistic', batch_size=5, max_iter=500)
    mlp.fit(train_x, train_y)
    return accuracy(test_y, mlp.predict(test_x))


def accuracy(y_true: object, y_pred: object) -> object:
    return np.mean(np.equal(y_true, y_pred))
