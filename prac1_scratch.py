#!/usr/bin/env python3

import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
import numpy as np

data = np.loadtxt('prac01data.csv', delimiter=',')
x_train, x_test, y_train, y_test = train_test_split(data[:, np.newaxis,0], data[:, np.newaxis, 1])

funs = [lambda x: x, lambda x: x**2, lambda x: x**3]

def add_cols(array, funs):
    new_array = list()
    for i in array:
        new_array.append(np.array([1, *list(map(lambda x: x(i[0]), funs))]))
    return np.array(new_array)

x_train = add_cols(x_train, funs)

x_test = add_cols(x_test, funs)

beta = reduce(np.matmul, [np.linalg.inv(np.matmul(np.transpose(x_train), x_train)), np.transpose(x_train), y_train])

predictions = np.matmul(sorted(x_test, key=lambda x:x[1]), beta)

plt.scatter(x_train[:,1], y_train)
plt.plot(sorted(x_test[:,1]), predictions, color="orange")
plt.show()
