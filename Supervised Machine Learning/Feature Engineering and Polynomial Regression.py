print ("=======================================================")
print ("-----------------------------------------------")

########################################################################################################################################################################################
########  explore feature engineering and polynomial regression which allows you to use the machinery of linear regression to fit very complicated, even very non-linear functions.########
########################################################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng ,plot_cost_i_w

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays



x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w  ,  model_b,_ = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("no feature engineering")
plt.plot(x,np.dot(X,model_w) + model_b, label="Predicted Value")
plt.xlabel("X"); plt.ylabel("y")
plt.legend()
plt.show()

print ("----------------------------------------------1")

x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature
X = X.reshape(-1, 1)  
model_w  ,  model_b , _ = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-5)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.show()

print ("----------------------------------------------2")


# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
print (X)
print (X.T)
print(X.T.shape)

model_w,model_b , _ = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.show()



print ("----------------------------------------------3")

# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 6), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()

print ("----------------------------------------------4")
##### Scaling features #####

# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)
print(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

x = np.arange(0,20,1)
y = x**2
X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b , hist = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.show()


print ("----------------------------------------------5")

##########  Complex Functions ¶  ##########

x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b,_ = run_gradient_descent_feng(X, y, iterations=100000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()


print ("=======================================================")
print ("-----------------------------------------------")
