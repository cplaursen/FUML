import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.loadtxt('prac01data.csv',delimiter=',')
X = data[:,np.newaxis, 0]
y = data[:,1]

#plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
#plt.show()

train_size = 70

train_X = X[:train_size]
train_y = y[:train_size]

test_X = X[train_size:]
test_y = y[train_size:]

m = LinearRegression().fit(train_X,train_y)
r2 = m.score(test_X,test_y)
print('Test set R^2 is', r2)

# create two new variables.
# so that we can fit a cubic function
X2 = X**2
X3 = X**3
# add them as columns (axis=1)
newX = np.concatenate((X,X2,X3),axis=1)

newtrain_X = newX[:train_size]
newtest_X = newX[train_size:]

m = LinearRegression().fit(newtrain_X,train_y)
r2 = m.score(newtest_X,test_y)
print('New test set R^2 is', r2)
