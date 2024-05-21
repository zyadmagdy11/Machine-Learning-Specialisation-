
print ("=======================================================")
print ("-----------------------------------------------")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from public_tests_a1 import * 
tf.keras.backend.set_floatx('float64')
from assigment_utils import *
tf.autograph.set_verbosity(0)



# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)

#split the data using sklearn routine 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)


"""
You can see below the data points that will be part of training (in red) are intermixed
with those that the model is not trained on (test).
This particular data set is a quadratic function with noise added.
The "ideal" curve is shown for reference.
"""

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()



def eval_mse(y, yhat):

    from sklearn.metrics import mean_squared_error

    m = len(y)
    err = 0.0
    err = mean_squared_error (y,yhat)/2
    return(err)



"""
Let's build a high degree polynomial model to minimize training error.
This will use the linear_regression functions from sklearn.
The code is in the imported utility file if you would like to see the details
"""



# create a model in sklearn, train on training data
degree = 10
lmodel = lin_model(degree)
lmodel.fit(X_train, y_train)

# predict on training data, find training error
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)

# predict on test data, find error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)


# plot predictions over data range 
x = np.linspace(0,int(X.max()),100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

"""
The following plot shows why this is
The model fits the training data very well.
To do so, it has created a complex function.
The test data was not part of the training and the model does a poor job of predicting on this data.
"""



# plot predictions over data range 
x = np.linspace(0,int(X.max()),100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)

"""
The test set error shows this model will not work well on new data.
If you use the test error to guide improvements in the model,
then the model will perform well on the test data...
but the test data was meant to represent new data.
You need yet another set of data to test new data performance.

The proposal made during lecture is to separate data into three groups.
The distribution of training, cross-validation and test sets
shown in the below table is a typical distribution,
but can be varied depending on the amount of data available.

Let's generate three data sets below.
We'll once again use train_test_split from sklearn
but will call it twice to get three splits:
"""

X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)

#split the data using sklearn routine 
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)

del X_ , y_

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_cv, y_cv,       color = dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()


"""
Let's train the model repeatedly,
increasing the degree of the polynomial each iteration
"""

max_degree = 9
err_train = np.zeros(max_degree)    
err_cv = np.zeros(max_degree)      
x = np.linspace(0,int(X.max()),100)  
y_pred = np.zeros((100,max_degree))  #columns are lines to plot

for degree in range(max_degree):
    lmodel = lin_model(degree+1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)

    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)

    y_pred[:,degree] = lmodel.predict(x)
    
optimal_degree = np.argmin(err_cv)+1

plt.close("all")
plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, 
                   err_train, err_cv, optimal_degree, max_degree)



""" Tuning Regularization

In previous labs, you have utilized regularization to reduce overfitting. 
Similar to degree, one can use the same methodology 
to tune the regularization parameter lambda (𝜆).

Let's demonstrate this by starting with a high degree polynomial
and varying the regularization parameter.
"""

degree = 10
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
num_steps = len(lambda_range)

err_train = np.zeros(num_steps)    
err_cv = np.zeros(num_steps)      
x = np.linspace(0,int(X.max()),100)  
y_pred = np.zeros((100,num_steps))  #columns are lines to plot

for i in range (num_steps):
    lmodel = lin_model(degree , regularization = True , lambda_ = lambda_range[i])
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)

    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)

    y_pred[:,i] = lmodel.predict(x)
    
optimal_reg_idx = np.argmin(err_cv) 

plt.close("all")
plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)





"""     Getting more data: Increasing Training Set Size 

When a model is overfitting (high variance),
collecting additional data can improve performance.
"""

X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)




"""     Evaluating a Learning Algorithm (Neural Network)
"""



X, y, centers, classes, std = gen_blobs()

# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)

del X_ , y_

plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)


""" Complex model """


def eval_cat_err(y, yhat):

    m = len(y)
    incorrect = 0
    for i in range(m):
        if y[i] != yhat[i]:
            incorrect += 1  
    cerr = incorrect/m
    
    return(cerr)

tf.random.set_seed(1234)
model = Sequential(
    [
        Dense (120 , activation = 'relu'  ,   name = 'layer1'),
        Dense (40 , activation = 'relu'  ,   name = 'layer2'),
        Dense (6 , activation = 'linear'  , name = 'layer3')
  
    ], name="Complex"
)
model.compile(
    loss= SparseCategoricalCrossentropy (from_logits=True),
    optimizer=Adam (0.01),
)

model.fit(
    X_train, y_train,
    epochs=100
)

model.summary()

#make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")

"""
This model has worked very hard to capture outliers of each category.
As a result, it has miscategorized some of the cross-validation data.
Let's calculate the classification error.
"""

training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")


""" Simple model  """


tf.random.set_seed(1234)
model_s = Sequential(
    [
        Dense (6 , activation = 'relu'  ,   name = 'layer1'),
        Dense (6 , activation = 'linear'  ,   name = 'layer2'),  
    ], name="Complex"
)
model_s.compile(
    loss= SparseCategoricalCrossentropy (from_logits=True),
    optimizer=Adam (0.01),
)

model_s.fit(
    X_train,y_train,
    epochs=100
)
model_s.summary()

model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")

training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
print ("-----------------------------------------------")
print ("=======================================================")
