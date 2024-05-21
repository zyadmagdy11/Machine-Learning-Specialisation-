import numpy as np

def load_data():
    data = np.loadtxt("/Users/magdyroshdy/Desktop/Python/Machine Learning/Practice Lab, Linear Regression/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

def load_data_multi():
    data = np.loadtxt("/Users/magdyroshdy/Desktop/Python/Practice Lab, Linear Regression/ex2data1.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

