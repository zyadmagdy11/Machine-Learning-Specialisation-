print ("=======================================================")
print ("-----------------------------------------------")

import numpy as np    # it is an unofficial standard to use np for numpy
import time

######################################################
################## LAB 1 ##################
######################################################
# a = np.random.random_sample(4)

# print (a)

# np.random.seed(1)

# a = np.random.rand(100) 
# b = np.random.rand(100)

# tic = time.time()  # capture start time
# print (tic)
# c = np.dot(a, b)
# toc = time.time()  # capture end time
# print (toc)

# print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")
# print ("-----------------------------------------------1")


# X = np.array([[1],[2],[3],[4]])
# w = np.array([2])
# c = np.dot(X[1], w)

# print (X)
# print (w)
# print (c)

# print ("-----------------------------------------------2")


# a = np.zeros((3, 5))                                       
# print(f"a shape = {a.shape}, a = {a}")                     

# a = np.zeros((2, 1))                                                                   
# print(f"a shape = {a.shape}, a = {a}") 

# a = np.random.random_sample((1, 1))  
# print(f"a shape = {a.shape}, a = {a}")
# print ("-----------------------------------------------3")


# #vector indexing operations on matrices
# a = np.arange(6).reshape(2,3)   #reshape is a convenient way to create matrices
# print(f"a.shape: {a.shape}, \na= {a}")
# print ("-----------------------------------------------4")
######################################################
################## LAB 2 ##################
######################################################


import copy, math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)                                                                            # reduced display precision on numpy arrays


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])                                   #[Size (sqft)  , Number of Bedrooms , Number of floors , Age of Home]
y_train = np.array([460, 232, 178])                                                                         # Price (1000s dollars)
print (X_train , "==>" , y_train.reshape(-1,))

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])                                   # For demonstration, w and b will be loaded with some initial selected values that are near the optimal
  
print ("----------------------------------------8")




def compute_cost(X, y, w, b): 

    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b          
        cost   +=  (f_wb_i - y[i]) **2       
    cost = cost / (2 * m)                      
    return cost


def compute_gradient(X, y, w, b): 
  
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] +=   err * X[i, j]    
        dj_db +=  err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, compute_gradient, alpha, num_iters): 
  
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        dj_db,dj_dw = compute_gradient(X, y, w, b)   

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing




print ("----------------------------------------8")


# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:f},{w_final} ")




################ plot cost versus iteration  ################
fig, ax1 = plt.subplots(1, 1, constrained_layout=False, figsize=(12, 4))
ax1.plot(J_hist)

ax1.set_title("Cost vs. iteration")
ax1.set_ylabel('Cost')             
ax1.set_xlabel('iteration step')   
plt.show()













print ("=======================================================")
print ("-----------------------------------------------")

