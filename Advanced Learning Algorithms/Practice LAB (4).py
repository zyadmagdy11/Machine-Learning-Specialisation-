
print ("=======================================================")
print ("-----------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils3 import *

"""
Suppose you are starting a company that grows and sells wild mushrooms.
Since not all mushrooms are edible, you'd like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
ou have some existing data that you can use for this task.
Can you use the data to help you identify which mushrooms can be sold safely?
"""

"""
X_train contains three features for each example

Brown Color (A value of 1 indicates "Brown" cap color and 0 indicates "Red" cap color)
Tapering Shape (A value of 1 indicates "Tapering Stalk Shape" and 0 indicates "Enlarging" stalk shape)
Solitary (A value of 1 indicates "Yes" and 0 indicates "No")
"""
"""
y_train is whether the mushroom is edible

y = 1 indicates edible
y = 0 indicates poisonous
"""


X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])





print ("-----------------------------------------------")
print ("=======================================================")
