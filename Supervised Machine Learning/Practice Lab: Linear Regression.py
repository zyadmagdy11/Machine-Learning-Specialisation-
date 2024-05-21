print ("=======================================================")
print ("-----------------------------------------------")


import numpy as np
import matplotlib.pyplot as plt
from utils import *
from lab_utils_multi import run_gradient_descent_feng , zscore_normalize_features
import copy
import math

#################### LAB CODE ##############################
############################################################


x_train, y_train = np.array (load_data() )             # x_train is the population of a city times 10,000 
# y_train represent your restaurant's average monthly profits in each city in units of $10,000. A negative value for profit indicates a loss.

def compute_cost(x, y, w, b): 
    m = x.shape[0] 
    
    total_cost = 0
    
    for i in range(m):
        total_cost += (( w*x[i] + b ) - y[i] )**2
    total_cost *= (1/2*m)

    return total_cost

def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    
    dj_dw = 0
    dj_db = 0
    
    
    for i in range(m):
        err = ( w*x[i] + b ) - y[i]
        dj_db += err
        dj_dw += err * x[i]
        
    dj_db *= 1/m
    dj_dw *= 1/m
    
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
  
    m = len(x)
    
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  
    b = b_in
    
    for i in range(num_iters):

        dj_dw, dj_db = gradient_function(x, y, w, b )  

        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        if i<100000:      
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    print (f"w,b found by gradient descent: w: {w}, b: {b}")    
    return w, b, J_history, w_history 



initial_w = 0.
initial_b = 0.


iterations = 1500
alpha = 0.01
w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)

m = x_train.shape[0]
predicted = np.zeros(m)

x = np.linspace(0,22,x_train.shape[0])
for i in range(m):
    predicted[i] = w * x[i] + b


plt.plot(x, predicted, c = "b" , label = "Predicted Line")
plt.scatter(x_train, y_train, marker='x', c='r' , label = "Actual Values") 

plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')

plt.legend()
plt.show()

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))








#################### MY CODE ##############################
############################################################

x_train, y_train = np.array (load_data() )   
x_train = x_train.reshape(-1,1)



plt.scatter(x_train , y_train , marker='x', c='r') 
plt.title("Profits vs. Population per city")
plt.xlabel ("Population * 10,000 ")
plt.ylabel ("Average Monthly Profits * $10,000")
plt.show()

w , b , h = run_gradient_descent_feng(x_train , y_train , iterations = 1500 , alpha =  0.01)


plt.scatter(x_train , y_train , marker='x', c='r', label="Actual Value")
plt.plot(x_train, np.dot(x_train,w) + b, label="Predicted Value")
plt.legend()
plt.show()


plt.plot (  h["cost"]  )
plt.title ("Cost Function Versus Number of iteration")
plt.xlabel ("Iterations")
plt.ylabel ("Cost Function")
plt.show()








print ("=======================================================")
print ("-----------------------------------------------")