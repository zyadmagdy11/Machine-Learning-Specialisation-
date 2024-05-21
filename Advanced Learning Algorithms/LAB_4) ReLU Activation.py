print ("=======================================================")
print ("-----------------------------------------------")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import linear, relu, sigmoid
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from autils_copy import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)



plt_act_trio()

_ = plt_relu_ex()












































print ("-----------------------------------------------")
print ("=======================================================")
