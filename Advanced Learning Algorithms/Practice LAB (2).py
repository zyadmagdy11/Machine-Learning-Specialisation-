
print ("=======================================================")
print ("-----------------------------------------------")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.activations import linear, relu, sigmoid
import matplotlib.pyplot as plt

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


from autils import *
from lab_utils_softmax import plt_softmax
np.set_printoptions(precision=2)





def my_softmax(z):  

    a = np.exp(z)/np.sum(np.exp(z))
    return a


z = np.array([1., 2., 3., 4.])
a = my_softmax(z)
atf = tf.nn.softmax(z)
print(f"my_softmax(z):         {a}")
print(f"tensorflow softmax(z): {atf}")


plt.close("all")
# plt_softmax(my_softmax)


X, y = load_data()


print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))




def plot_digits (X,y):
    m, n = X.shape
    fig, axes = plt.subplots(8,8, figsize=(5,5))
    fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.91]) #[left, bottom, right, top]
    for i,ax in enumerate(axes.flat):
        random_index = np.random.randint(m)
        
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        ax.imshow(X_random_reshaped, cmap='gray')
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
        fig.suptitle("Label, image", fontsize=14)
    plt.show()

plot_digits (X,y)


tf.random.set_seed(1234) 
model = Sequential(
    [               
        ### START CODE HERE ### 
        tf.keras.Input(shape=(400,)),

        Dense (units = 25 , activation = 'relu' , name = 'layer1'),
        Dense (units = 15 , activation = 'relu' , name = 'layer2'),
        Dense (units = 10 , activation = 'linear' , name = 'layer3')
    
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)

[l1 , l2 , l3] = model.layers



model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

model.fit(
    X,y,
    epochs=40
)





image_of_two = X[1015]


prediction = model.predict(image_of_two.reshape(1,400))  # prediction

print(f" predicting a Two: \n{prediction}")
print(f" Largest Prediction index: {np.argmax(prediction)}")


prediction_p = tf.nn.softmax(prediction)

print(f" predicting a Two. Probability vector: \n{prediction_p}")
print(f"Total of predictions: {np.sum(prediction_p):0.3f}")


# Select the most argument number , (errors)













































print ("-----------------------------------------------")
print ("=======================================================")
