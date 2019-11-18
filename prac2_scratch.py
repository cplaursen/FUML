#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('breast-cancer-wisconsin.data',delimiter=',',missing_values='?')
data = data[~np.isnan(data).any(axis=1)]


