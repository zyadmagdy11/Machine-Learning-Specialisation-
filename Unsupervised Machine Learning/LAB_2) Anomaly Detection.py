print ("=======================================================")
print ("-----------------------------------------------")


"""
In this exercise, you will implement an anomaly detection algorithm to detect anomalous behavior in server computers.

The dataset contains two features -
    throughput (mb/s) and
    latency (ms) of response of each server.
While your servers were operating, you collected  𝑚=307 examples of how they were behaving
You suspect that the vast majority of these examples are “normal” (non-anomalous) examples of the servers operating normally,
but there might also be some examples of servers acting anomalously within this dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils2 import *

X_train, X_val, y_val = load_data()


# Visualize your data

plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

plt.title("The first dataset")
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()

""" To perform anomaly detection, you will first need to fit a model to the data’s distribution (Gaussian distribution)"""



def estimate_gaussian(X): 
   
    
    mu = np.mean(X , axis = 0)
    var = (np.std(X , axis = 0))**2   

    return mu, var


def multivariate_gaussian_mywork(X , mu , var):

    p = (1/( np.sqrt(2* np.pi) * np.sqrt(var) )) * np.exp( -0.5 * ((X_train-mu)**2) /var)
    p = np.product(p , axis = 1) 
    return p




mu, var = estimate_gaussian(X_train)              

print("Mean of each feature:", mu)
print("Variance of each feature:", var)
    

P1 = multivariate_gaussian(X_train, mu, var)

P2 = multivariate_gaussian_mywork(X_train, mu, var)           # P1 = P2

# visualize_fit(X_train, mu, var)



"""
Selecting the threshold  𝜖
In this section, you will complete the code in select_threshold to select the threshold 𝜀
using the  F1 score on a cross validation set.
"""


def select_threshold(y_val, p_val): 

    step_size = (max(p_val) - min(p_val)) / 1000

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        p = (p_val<epsilon)
        tp = np.sum ((y_val == 1) & (p == 1) )
        fp = np.sum ((y_val == 0) & (p == 1) )
        fn = np.sum ((y_val == 1) & (p == 0) )

        prec = tp/(tp+fp)
        rec  = tp/(tp+fn)
        
        
        F1 = (2 * prec * rec)/(prec +rec)
        print (F1)
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1



p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)

print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)
    
"""
Now we will run your anomaly detection code and circle the anomalies in the plot 
"""


outliers = P1 < epsilon

visualize_fit(X_train, mu, var)

plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',markersize= 10,markerfacecolor='none', markeredgewidth=2)



"""
Now, we will run the anomaly detection algorithm that you implemented
on a more realistic and much harder dataset.

In this dataset, each example is described by 11 features,
capturing many more properties of your compute servers.
"""


X_train_high, X_val_high, y_val_high = load_data_multi()

"""
Now, let's run the anomaly detection algorithm on this new dataset.

The code below will use your code to
"""



mu_high, var_high = estimate_gaussian(X_train_high)

p_high = multivariate_gaussian(X_train_high, mu_high, var_high)

p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)

epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)

print('Best epsilon found using cross-validation: %e'% epsilon_high)
print('Best F1 on Cross Validation Set:  %f'% F1_high)
print('# Anomalies found: %d'% sum(p_high < epsilon_high))






print ("=======================================================")
print ("-----------------------------------------------")

