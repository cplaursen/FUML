#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

def performance():
    data = np.loadtxt("pca_ex.csv", delimiter=",")
    classes = np.loadtxt("classes.txt")
    X_train, X_test, y_train, y_test = train_test_split(data, classes)
    cmap = plt.cm.get_cmap("bwr")
    PCA_model = PCA(2)
    PCA_cols = PCA_model.fit_transform(X_train)
    PCA_test = PCA_model.transform(X_test)

    X12R, X23R, X13R, PCA_c = (LogisticRegression() for _ in range(4))

    def regPerformance(model, name, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        print(f"{name} got a score of {model.score(X_test, y_test)}")
        return model.predict(X_test)
    
    X12p = regPerformance(X12R, "X12R", X_train[:,:2], X_test[:,:2], y_train, y_test)
    X23p = regPerformance(X23R, "X23R", X_train[:,1:], X_test[:,1:], y_train, y_test)
    X13p = regPerformance(X13R, "X13R", np.delete(X_train,1,axis=1), np.delete(X_test,1,axis=1), y_train, y_test)
    PCAp = regPerformance(PCA_c, "PCA_c", PCA_cols, PCA_model.transform(X_test), y_train, y_test)
    fig, axes = plt.subplots(2,4)
    axes[0,0].set_title('X1 vs X2')
    axes[0,0].scatter(X_test[:,0], X_test[:,1] , c=y_test, cmap=cmap)
    axes[1,0].scatter(X_test[:,0], X_test[:,1] , c=X12p)
    axes[0,1].set_title('X1 vs X3')
    axes[0,1].scatter(X_test[:,0], X_test[:,2] , c=y_test, cmap=cmap)
    axes[1,1].scatter(X_test[:,0], X_test[:,2] , c=X23p)
    axes[0,2].set_title('X2 vs X3')
    axes[0,2].scatter(X_test[:,1], X_test[:,2] , c=y_test, cmap=cmap)
    axes[1,2].scatter(X_test[:,1], X_test[:,2] , c=X13p)
    axes[0,3].set_title('First 2 principal components')
    axes[0,3].scatter(PCA_test[:,0], PCA_test[:,1], c=y_test, cmap=cmap)
    axes[1,3].scatter(PCA_test[:,0], PCA_test[:,1], c=PCAp)
    plt.show()


def main():
    data = np.loadtxt("pca_ex.csv", delimiter=",")
    classes = np.loadtxt("classes.txt")
    PCA_cols = PCA(2).fit_transform(data) 
    cmap = plt.cm.get_cmap("bwr")
    fig, axes = plt.subplots(2,2)
    axes[0,0].set_title('X1 vs X2')
    axes[0,0].scatter(data[:,0], data[:,1] , c=classes, cmap=cmap)
    axes[1,0].set_title('X1 vs X3')
    axes[1,0].scatter(data[:,0], data[:,2] , c=classes, cmap=cmap)
    axes[0,1].set_title('X2 vs X3')
    axes[0,1].scatter(data[:,1], data[:,2] , c=classes, cmap=cmap)
    axes[1,1].set_title('First 2 principal components')
    axes[1,1].scatter(PCA_cols[:,0], PCA_cols[:,1], c=classes, cmap=cmap)
    plt.show()

def plot_3d():
    data = np.loadtxt("pca_ex.csv", delimiter=",")
    classes = np.loadtxt("classes.txt")
    ata = PCA().fit_transform(data, classes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=classes)
    ax.scatter(ata[:,0], ata[:,1], ata[:,2], c=classes, cmap=plt.cm.get_cmap("bwr"))
    plt.show()


if __name__ == "__main__":
    performance()
