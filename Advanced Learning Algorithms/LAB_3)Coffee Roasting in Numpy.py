
print ("=======================================================")
print ("-----------------------------------------------")

import numpy as np
import copy
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_utils_multi import run_gradient_descent_feng
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X,Y = load_coffee_data()




# Normalize Data
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

print (Xn)

g = sigmoid

def my_dense(a_in, W, b):
                         
    z = np.matmul (a_in , W) + b         
    a_out = g(z)               
    return(a_out)

def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)


def my_predict(X, W1, b1, W2, b2):          # same as model.predict()
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array([ [-9.82, -9.28,  0.96] ] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [[15.41]] )





X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

netf= lambda x : my_predict(norm_l(x),W1_tmp, b1_tmp, W2_tmp, b2_tmp)
plt_network(X,Y,netf)









##################################################################################################
# def not_function_in_LAB (X , Y):
#     def sigmoid(z):

#         g = 1/(1+np.exp(-z))
        
#         return g

#     def compute_cost(X, y, w, b, *argv):
    

#         m, n = X.shape
        
#         z_i = X @ w + b                                 
#         f_wb_i = sigmoid(z_i) 
#         total_cost =  np.sum ( -y*np.log(f_wb_i) - (1-y)*np.log(1-f_wb_i)  )/m                                                                   
                                                                        
#         return total_cost

#     def compute_gradient(X, y, w, b, *argv): 
    
#         m,n = X.shape
#         dj_dw = np.zeros((n,))
#         dj_db = 0.
        
#         z = X @ w + b
#         err = sigmoid(z) - y
#         dj_dw =  (1/m)  *  (   X.T @ err   )
#         dj_db =  (1/m)  *     np.sum(err)

#         return dj_db, dj_dw

#     def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
     
        
#         m = len(X)
        
#         J_history = []
#         w_history = []
        
#         for i in range(num_iters):

#             dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

#             w_in = w_in - alpha * dj_dw               
#             b_in = b_in - alpha * dj_db              
        
#             if i<100000:      # prevent resource exhaustion 
#                 cost =  cost_function(X, y, w_in, b_in, lambda_)
#                 J_history.append(cost)

#             if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
#                 w_history.append(w_in)
#                 print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
            
#         return w_in, b_in, J_history, w_history #return w and J,w history for graphing


#     initial_w = np.zeros (X.shape[1])
#     initial_b = 0

#     iterations = 10000
#     alpha = 0.09

#     w,b, J_history,_ = gradient_descent(X ,Y.reshape(-1,), initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)
#     print ("Waight =",w , "bias =" ,b)

#     #Check Cost Function Is decreasing
#     ###################################
#     plt.plot(range(iterations ) , J_history )
#     plt.ylabel("Cost Function")
#     plt.xlabel("#Iterations")
#     plt.show()
#     ####################################

# not_function_in_LAB (Xn.numpy(),Y)
##################################################################################################
print ("-----------------------------------------------")
print ("=======================================================")
