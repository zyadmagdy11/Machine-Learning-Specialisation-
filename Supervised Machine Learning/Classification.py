print ("=======================================================")
print ("-----------------------------------------------")


import copy, math
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data , draw_vthresh , sigmoid , dlc,  compute_cost_logistic , plt_tumor_data , gradient_descent
from plt_one_addpt_onclick import plt_one_addpt_onclick
from plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from plt_logistic_loss import soup_bowl, plt_logistic_squared_error
from plt_quad_logistic import plt_quad_logistic, plt_prob

from utils import *

from sklearn.linear_model import SGDRegressor


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])       #<<-
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train == 1         #[False , False , False , True , True, True ]
neg = y_train == 0         #[True , True , True , False , False, False ]

fig,ax = plt.subplots(1,2,figsize=(12,6))

ax[0].scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
ax[0].scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', edgecolors=dlc["dlblue"],lw=3)

ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y', fontsize=12)
ax[0].set_xlabel('x', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()



plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()

plt.show()



# This plot shows that linear regression is insufficient for classification problems
w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)
plt.show()



print ("################### Logistic Regression ###################")
print ("###########################################################")


input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# Input is a single number
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)


z_tmp = np.arange(-10,11)
y = sigmoid(z_tmp)
print("Input (z),  Output (sigmoid(z))")
print(np.c_[z_tmp, y])

fig,ax = plt.subplots(1,1,figsize=(12,6))
ax.plot(z_tmp, y, c="b")
ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
plt.show()


x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
plt.show()





print ("###################  Decision Boundary  ###################")
print ("###########################################################")

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

# z = w1x1 + w2x2 + b
# 𝑓(𝐱)=𝑔(−3+𝑥0+𝑥1)      ==>    w1 = 1  ,  w2 = 1  , b = -3  
#𝑦=1 if −3+𝑥0+𝑥1 >= 0       else y=0

# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1,alpha=0.2,color='g')

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()


# by using higher order polynomial terms (eg:  𝑓(𝑥)=𝑔(𝑥^2+𝑥1−1) ), we can come up with more complex non-linear boundaries.



print ("###################   Logistic Loss  ###################")
print ("########################################################")


soup_bowl()

x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train)
plt.show()

plt.close('all')
plt_logistic_squared_error(x_train,y_train)
plt.show()

plt_two_logistic_loss_curves()

plt.close('all')
cst = plt_logistic_cost(x_train,y_train)




print ("###################   Cost Function for Logistic Regression  ###################")
print ("################################################################################")


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) 
y_train = np.array([0, 0, 0, 1, 1, 1])                                           

fig,ax = plt.subplots(1,1,figsize=(8,6))
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=20)
ax.set_xlabel('$x_0$', fontsize=20)
plt.show()
plt.show()


def compute_cost_logistic(X, y, w, b):

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))


fig , ax = plt.subplots(1,1 , figsize = (12,6))



x0 = np.arange(0,6)
x1 = 3-x0
x1_other = 4-x0

plot_data(X_train, y_train, ax)

ax.plot(x0 , x1 , c = "b"  , label = "b = -3")
ax.plot(x0 , x1_other , c = "m" , label = "b = -4"  )

ax.axis ([0,4 , 0,4])

ax.set_xlabel("$X_0$")
ax.set_ylabel("$X_1$")
plt.title("Decision Boundary")

plt.legend()
plt.show()


# Which one from (w1,b1) (w1,b2) is Optimum choice 
w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))






print ("###################   Gradient descent for logistic regression  ###################")
print ("###################################################################################")

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.array([2.,3.])
b_tmp = 1.



def compute_gradient_logistic(X, y, w, b):
    
    m,n = X.shape

    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        f_wb = sigmoid(  np.dot(X[i],w) + b ) - y[i]
        dj_db += f_wb
        for j in range(n):
            dj_dw[j] += f_wb * X[i,j]

    dj_dw = dj_dw/m
    dj_db = dj_db/m 
    return dj_db , dj_dw    


def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    m,n = X.shape

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db , dj_dw = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w , b , J_history        


w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, J = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
   

#Check Cost Function Is decreasing
###################################
plt.plot(range(iters) , J )
plt.xlabel("Cost Function")
plt.ylabel("#Iterations")
plt.show()
####################################

fig,ax = plt.subplots(1,1,figsize=(5,4))

ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()



x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()

w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )
plt.show()




print ("######################################   Overfitting  ######################################")
print ("############################################################################################")


from ipywidgets import Output 
from plt_overfit import overfit_example, output


##########Generate a plot that will allow you to explore overfitting##########
plt.close("all")
ofit = overfit_example(False)
plt.show()
#############################################################################



print ("##############################   Regularized Cost and Gradient  ############################")
print ("############################################################################################")


def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
  
    m,n  = X.shape
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   
        cost = cost + (f_wb_i - y[i])**2                                            
    cost = cost / (2 * m)                                                
 
    reg_cost = (lambda_/(2*m)) * np.sum(w**2)
                           
    total_cost = cost + reg_cost                                       
    return total_cost                                                  


def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):

    m,n = X.shape

    z_i = X @ w + b                                 
    f_wb_i = sigmoid(z_i) 

    cost =  np.sum ( -y*np.log(f_wb_i) - (1-y)*np.log(1-f_wb_i)  )/m                                                                   
    reg_cost = (lambda_/(2*m)) * np.sum(w**2)                                         
        
    total_cost = cost + reg_cost                                       
    return total_cost


def compute_gradient_linear_reg(X, y, w, b, lambda_):

    m,n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0.

    err = (X @ w + b) - y
    dj_dw =  (1/m) *  (   X.T @ err   ) + (lambda_/m) * w
    dj_db = (1/m) * np.sum(err)

    return dj_db, dj_dw


def compute_gradient_logistic_reg(X, y, w, b, lambda_):

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    err = sigmoid(X @ w + b) - y
    dj_dw =  (1/m) *  (   X.T @ err   ) + (lambda_/m) * w
    dj_db = (1/m) * np.sum(err)

    return dj_db, dj_dw

gradient_descent



print ("##############################   Practice Lab on Logistic Regression  ############################")
print ("##################################################################################################")

 
X_train, y_train = load_data_multi()

print (X_train)
print (X_train.shape)

print(y_train)
print(y_train.shape)

def plot_data_prac(X, y, s=30 , labe1 = "" , label2 = "" ):

    
    pos = y == 1        
    neg = y == 0
    pos = pos.reshape(-1,)  
    neg = neg.reshape(-1,)

    # Plot examples
    fig , ax = plt.subplots(1,1 , figsize = (12,6))
    ax.scatter(X[pos, 0], X[pos, 1], marker='+', s=s, c = 'k', label=labe1)
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', s=s, c = 'y' ,label=label2)
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')
    plt.legend()
    plt.show() 

plot_data_prac(X_train , y_train , labe1="Admitted" , label2= "Not Admitted")


def sigmoid_prac(z):
    g = 1/(1+np.exp(-z))
    return g


def compute_cost_prac(X, y, w, b, *argv):

    m, n = X.shape

    z_i = X @ w + b                                 
    f_wb_i = sigmoid(z_i) 

    total_cost =  np.sum ( -y*np.log(f_wb_i) - (1-y)*np.log(1-f_wb_i)  )/m                                                                   

    return total_cost



def compute_gradient_prac(X, y, w, b, *argv): 
   
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    z = X @ w + b
    err = sigmoid(z) - y
    dj_dw =  (1/m)  *  (   X.T @ err   )
    dj_db =  (1/m)  *     np.sum(err)

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 


    m = len(X)

    J_history = []
    w_history = []
    
    for i in range(num_iters):

        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        
        if i<100000:     
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 



def predict(X, w, b): 
  

    m, n = X.shape   
    p = np.zeros(m)
   
    z_wb =   X @ w  + b
    f_wb = sigmoid(z_wb)
    p = f_wb >= 0.5
        
    return p


w = np.zeros(X_train.shape[1])
b = 0
p = predict(X_train, w,b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))




######   Regularized Logistic Regression  ###### 
###### ###### ###### ###### ###### ###### ###### 

X_train, y_train = load_data()

plot_data_prac(X_train , y_train , labe1="Accepted",  label2 = "Rejected")

print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)



print ("=======================================================")
print ("-----------------------------------------------")

