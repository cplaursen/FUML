#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np 

california = fetch_california_housing()
X, y = (california[:,:-1], california[:,-1])
X = np.insert(X, 0, 1, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)

beta = reduce(np.matmul, [np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train), y_train])
