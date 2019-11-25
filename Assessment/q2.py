#!/usr/bin/env python3

import sys
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split

def main():
    data = np.loadtxt('continuous.csv', delimiter=',')
    testData = np.loadtxt('secrettest.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    X_test = testData[:, :-1]
    y_test = testData[:, -1]
    models = [("Lasso", LassoCV()), ("Ridge", RidgeCV()), ("Linear Regression", LinearRegression())]
    for name, model in models:
        model.fit(X, y)
        print(f"{name} scored {model.score(X_test, y_test)}")


if __name__ == "__main__":
    main()
