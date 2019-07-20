import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

file_name = 'Data_Classification_7.csv'
raw_data = pd.read_csv(file_name).as_matrix()
m, n = raw_data.shape
X = raw_data[:, 0:n-1]
Y = raw_data[:, n-1]
Y = Y.reshape(-1,1)
x = X.ravel()
y_old = Y.ravel()

X = np.delete(X, 8, 1)
X = np.delete(X, 11, 1)
X = np.delete(X, 11, 1)
X = np.delete(X, 16, 1)
X = np.delete(X, 16, 1)
row, column = X.shape

ones = np.ones(shape=(X.shape[0], 1))

X = np.concatenate((ones, X), axis=1)
for i in range(1, X.shape[1]):
        X[:, i] = (X[:, i] - np.amin(X[:, i]))/(np.amax(X[:, i]) - np.amin(X[:, i]))

W = np.array([np.random.rand(column+1)])

W = W.T


X = np.matrix(X)
Y = np.matrix(Y)
W = np.matrix(W)

iterations = []
errors = []

alpha = 0.1
fold_no = 10;
Folds = KFold(n_splits=fold_no)
Folds.get_n_splits(X)
average_accuracy = 0

for train_index,test_index in Folds.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    MaxIter = 100
    Iter = 0
    while Iter <= MaxIter:
        y = X_train * W
        y = 1/(1+np.exp(-1*y))
        r, c = y.shape
        for i in range(r):
            if y[i] >= .5:
                y[i] = 1
            else:
                y[i] = 0
        E = y - Y_train
        L = X.T

        for i in range(n):
            S = L[:, i] * E[i]
            W = W - alpha * S
        Iter += 1
    y = X_test * W
    y = 1 / (1 + np.exp(-1 * y))
    r, c = y.shape
    for i in range(r):
        if y[i] >= .5:
            y[i] = 1
        else:
            y[i] = 0
    count = 0
    for i in range(r):
        if y[i] - Y_test[i] == 0:
            count += 1

    accuracy = (count /r) * 100
    average_accuracy += accuracy
    print( accuracy, "%")

print("average_accuracy",average_accuracy/10, "%")