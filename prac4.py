#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = np.genfromtxt('breast-cancer-wisconsin.data',delimiter=',',missing_values='?')
data = data[~np.isnan(data).any(axis=1)]

X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1])

PCA_model = PCA()

PCA_cols = PCA_model.fit_transform(X_train)
print(PCA_model.components_)

regNormal = LogisticRegression()
regNormal.fit(X_train, y_train)
print(f"Score without PCA: {regNormal.score(X_test, y_test)}")
regPCA = LogisticRegression() 
regPCA.fit(PCA_cols, y_train)
print(f"Score with PCA: {regPCA.score(PCA_model.transform(X_test), y_test)}")

print(PCA_cols)

plt.scatter(PCA_cols[:,1], PCA_cols[:,2], c=y_train)
plt.show() 
