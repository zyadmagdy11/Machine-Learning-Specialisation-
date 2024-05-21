print ("=======================================================")
print ("-----------------------------------------------")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils2 import *

# _ = plot_entropy()
# plt.show()

X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


"""
On each node, we compute the information gain for each feature,
then split the node on the feature with the higher information gain,
by comparing the entropy of the node with the weighted entropy in the two splitted nodes.
"""

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    
print(entropy(0.5))

"""
To illustrate, let's compute the information gain if we split the node for each of the features. 
To do this, let's write some functions.
"""

def split_indices(X, index_feature):
    """
    Given a dataset and a index feature,
    return two lists for the two split nodes,
    the left node has the animals that have that feature = 1
    and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

split_indices(X_train, 0)

def weighted_entropy(X,y,left_indices,right_indices):

    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p1_left = sum (y[left_indices])/ len(left_indices)
    p1_right = sum (y[right_indices])/ len(right_indices)

    return w_left * entropy(p1_left) + w_right * entropy(p1_right)

left_indices, right_indices = split_indices(X_train, 0)
print(weighted_entropy(X_train, y_train, left_indices, right_indices))        


def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return h_node - w_entropy

information_gain(X_train, y_train, left_indices, right_indices)

"""
Now, let's compute the information gain if we split the root node for each feature:
"""

for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")




print ("-----------------------------------------------")
print ("=======================================================")
