print ("=======================================================")
print ("-----------------------------------------------")


import numpy as np
import matplotlib.pyplot as plt
import random
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl


################## LAB #########################


x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])


def compute_cost(x_train,y_train, w): 
    m = x_train.shape[0] 
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x_train[i]  
        cost = (f_wb - y_train[i]) ** 2  
        cost_sum = cost_sum + cost     
    total_cost = (1 / (2 * m)) * cost_sum  
    return total_cost


plt_intuition(x_train,y_train)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
plt.show()

soup_bowl()
plt.show()


















############ My Solustion ############

# x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
# y_train = np.array([250, 300, 480,  430,   630, 730,])

# def compute_cost(x_train,y_train, w): 
#     m = x_train.shape[0] 
#     cost_sum = 0 
#     for i in range(m): 
#         f_wb = w * x_train[i]  
#         cost = (f_wb - y_train[i]) ** 2  
#         cost_sum = cost_sum + cost     
#     total_cost = (1 / (2 * m)) * cost_sum  
#     return total_cost

# J_w = np.zeros(200000)
# J_i = np.zeros(200000)
# c = 0
# for i in range (-100000,100000,1):
#     J_w [c] = compute_cost(x_train,y_train  , i )
#     J_i[c]  = i
#     c += 1


# w = J_i[np.where(J_w == np.amin(J_w))]
# print (w,"=>",np.amin(J_w))

# #Plotting Straight Line and points dataset 
# x = np.array([-100, 100])
# y = w * x 
# plt.plot(x, y)
# plt.scatter(x_train , y_train , marker='*' )
# plt.grid(True)
# plt.show()
    

# print(J_w)
# plt.plot(J_i,J_w , 'o')
# plt.grid(True)
# plt.show()


print ("=======================================================")
print ("-----------------------------------------------")
