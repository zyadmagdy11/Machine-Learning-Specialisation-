print ("=======================================================")
print ("-----------------------------------------------")

import math, copy
import numpy as np
from numpy import random

import matplotlib.pyplot as plt
from lab_utils_multi import run_gradient_descent , load_data_multi



X_train = random.randint(100000 , size = (99,4))
X_features = ['size(sqft)','bedrooms','floors','age']

y_train = np.array ([[300. ,  509.8,  394.,   540.,   415.,   230.,   560.,   294.,   718.2,  200.,
 302.,   468.,   374.2 , 388.,   282. ,  311.8 , 401. ,  449.8  ,301.  , 502.,
 340. ,  400.28 ,572.,   264.,   304. ,  298. ,  219.8 , 490.7 , 216.96 ,368.2,
 280.,   526.87, 237.,   562.43 ,369.8,  460. ,  374. ,  390. ,  158. ,  426.,
 390.,   277.77, 216.96, 425.8 , 504. ,  329. ,  464. ,  220. ,  358.,   478.,
 334.,   426.98 ,290.,   463.,   390.8 , 354. ,  350. ,  460. ,  237. ,  288.3,
 282.,   249. ,  304. ,  332. ,  351.8  ,310. ,  216.96, 666.34 ,330.  , 480.,
 330.3 , 348. ,  304. ,  384. ,  316. ,  430.4 , 450. ,  284. ,  275. ,  414.,
 258.,   378. ,  350. ,  412. ,  373.  , 225. ,  390. ,  267.4 , 464. ,  174.,
 340. ,  430. ,  440. ,  216. ,  329. ,  388. ,  390. ,  356. ,  257.8 ]])
################  Mean Normalisation ################
#####################################################

mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma 


# fig,ax=plt.subplots(1, 1, figsize=(12, 5) , sharey=True)

# ax.scatter(X_norm[:,0], X_norm[:,3])
# ax.set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
# ax.set_title(r"Z-score normalized")
# ax.axis('equal')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.suptitle("distribution of features before, during, after normalization")
# plt.show()


w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 100000, 1.0e-1, )

x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")





























print ("=======================================================")
print ("-----------------------------------------------")