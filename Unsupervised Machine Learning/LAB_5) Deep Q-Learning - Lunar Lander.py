print ("=======================================================")
print ("-----------------------------------------------")

import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils4

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# Set up a virtual display to render the Lunar Lander environment.
Display(visible=0, size=(840, 480)).start()

# Set the random seed for TensorFlow
tf.random.set_seed(utils4.SEED)


MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate  
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps







print ("=======================================================")
print ("-----------------------------------------------")

