print ("=======================================================")
print ("-----------------------------------------------")



import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2])
y_train = np.array([300, 500])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")


plt.scatter(x_train, y_train, marker='x', c='r')
plt.plot(x_train,y_train)
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()





print ("=======================================================")
print ("-----------------------------------------------")