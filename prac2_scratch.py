#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import fsolve


data = np.genfromtxt('breast-cancer-wisconsin.data',delimiter=',',missing_values='?')
data = data[~np.isnan(data).any(axis=1)]

data[:,-1] = np.array(list(map(lambda x: 0 if x == 2 else 1, data[:,-1])))

X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1])


def p(x, betas): # Pr(y=1 | X=x, beta)
    return 1/(1+np.exp(np.matmul(np.transpose(betas), x)))


def logistic_regression(X, y):
    """Compute binary logistic regression coefficients

    :X: nxm numpy matrix of predictors
    :y: nx1 column matrix of classes, either 0 or 1
    :returns: n+1 length array of coefficients

    """
    new_X = np.insert(X, 0, 1, axis=1) # Add bias 


    def score_equations(betas):
        sum_ = 0
        for n, i in enumerate(new_X):
            sum_ += i * (y[n] - p(i, betas))
        return sum_

    return fsolve(score_equations, np.zeros(new_X.shape[1]))

def logistic_fit(betas, X):
    new_X = np.insert(X, 0, 1, axis=1) # Add bias 
    return np.array([(p(i, betas), 1, y_test[n]) if p(i, betas) > 0.5 else (p(i, betas),0, y_test[n]) for n, i in enumerate(new_X)])


print(logistic_fit(logistic_regression(X_train, y_train), X_test))
