import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import plt_stationary
from sklearn.linear_model import LinearRegression, SGDRegressor

from numpy import random

x_train = np.array([[1.0,1.22,1.25,1.56,1.87,1.99, 2.0,2.4,2.8,3.2,3.6]])
y_train = np.array([475.0, 500.0,525,550,575,600,625,650,675,700,725])


model = SGDRegressor()
model.fit (x_train.reshape(11,1) , y_train)

w = model.coef_
b = model.intercept_
print ("W = ",model.coef_ , "\t b = ",model.intercept_ , "Number of iteration = " , model.n_iter_)

# x = x_train + 4 * x_train*(np.random.sample((11,))-0.5)
# yhat = model.predict (x.reshape(11,1))


plt.plot (np.arange(5) ,  w * np.arange(5) + b  , lw = 2)
plt.scatter(x_train , y_train , marker='x' , c='red',s=70)
plt.xlabel ("Size m$2$")
plt.ylabel ("Price in $")
plt.grid()
plt.show()

plt_stationary (x_train , y_train)




w1 = np.linspace (0, 500 , 3000)
b1 = np.linspace (0 , 500 , 3000)

w,b = np.meshgrid (w1,b1)




print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

def compute_model_output(x, w, b, y):

    m = x.shape[0]
    f_wb = np.zeros_like(w)
    sum = 0
    w_min = 0
    b_min = 0
    min = 1e30
    

  
    for i in range(w.shape[0]):
      for j in range(b.shape[1]):
        for n in range(m):
          sum += (w[i,j] * x[n] + b[i,j] - y[n])**2

        f_wb[i,j] = sum/(2*m)
        sum = 0
        if (f_wb[i,j] <= min):
          min = 1/(2*m) * f_wb[i,j]
          w_min = w[i,j]
          b_min = b[i,j]

      
    return f_wb , w_min , b_min , min



J , w_min , b_min , min = compute_model_output(x_train, w, b , y_train)

print ("minimum cost value =",min , "At w =",w_min , " ,b =",b_min)


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w, b, J, cmap = "Spectral_r", alpha=0.7, antialiased=False)
ax.plot_wireframe(w, b, J, color='k', alpha=0.1)
ax.set_xlabel("$w$")
ax.set_ylabel("$b$")
ax.set_zlabel("$J(w,b)$", rotation=90)
ax.set_title("$J(w,b)$\n [You can rotate this figure]", size=15)
plt.show()


plt.scatter(x_train , y_train , marker='x' , c='red',s=70 , label ="Actual Value")
plt.plot (x_train , w_min * x_train + b_min , c = 'b' , label = "Our Prediction")
plt.xlabel ("Size m$2$")
plt.ylabel ("Price in $")
plt.legend()
plt.grid()
plt.show()


