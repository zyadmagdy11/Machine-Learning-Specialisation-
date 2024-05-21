
print ("=======================================================")
print ("-----------------------------------------------")


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from autils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


X, y = load_data()

print (X)
print (X.shape)
print (y)
print (y.shape)

print (X[0].reshape(20,20))


#################### Visualizing the Data #########################################
###################################################################################
m, n = X.shape

fig, axes = plt.subplots(8,8, figsize=(8,8))
fig.tight_layout(pad=0.1)

for i,ax in enumerate(axes.flat):

    random_index = np.random.randint(m)
    
    X_random_reshaped = X[random_index].reshape((20,20)).T
    
    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')
    
    ax.set_title(y[random_index,0])
    ax.set_axis_off()
plt.show()
###################################################################################

model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(units = 25, activation='sigmoid', name = 'layer1'),
        Dense(units = 15, activation='sigmoid', name = 'layer2'),
        Dense(units = 1, activation='sigmoid', name = 'layer3')
    ]
)


model.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer=tf.keras.optimizers.Adam(0.001),)
model.fit(X,y,epochs=20)



def Examine_Weights_shapes(model):
    [layer1, layer2, layer3] = model.layers
    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")
Examine_Weights_shapes (model)

def Examine_Weights_Values(model):
    [layer1, layer2, layer3] = model.layers
    W1,b1 = layer1.get_weights()
    W2,b2 = layer2.get_weights()
    W3,b3 = layer3.get_weights()
    print(f"W1 = {W1},\n b1 = {b1}\n")
    print(f"W2 = {W2},\n b2 = {b2}\n")
    print(f"W3 = {W3},\n b3 = {b3}\n")
Examine_Weights_Values(model)


prediction = model.predict(X)  # a zero
Yhat = (prediction >= 0.5).astype(int)
print (Yhat)



# one of the misclassified images looks.
fig = plt.figure(figsize=(1, 1))
errors = np.where(y != Yhat)
random_index = errors[0][0]
X_random_reshaped = X[random_index].reshape((20, 20)).T
plt.imshow(X_random_reshaped, cmap='gray')
plt.title(f"{y[random_index,0]}, {Yhat[random_index, 0]}")
plt.axis('off')
plt.show()











print ("-----------------------------------------------")
print ("=======================================================")
