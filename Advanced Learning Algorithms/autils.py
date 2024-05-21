import numpy as np

def load_data():
    X = np.load("/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/Practice LAB (2)/X.npy")
    y = np.load("/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/Practice LAB (2)/y.npy")
    X = X[0:5000]               # Practice LAB (1): X = X[0:1000]
    y = y[0:5000]               # Practice LAB (1): y = y[0:1000]
    return X, y

def load_weights():
    w1 = np.load("data/w1.npy")
    b1 = np.load("data/b1.npy")
    w2 = np.load("data/w2.npy")
    b2 = np.load("data/b2.npy")
    return w1, b1, w2, b2

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
