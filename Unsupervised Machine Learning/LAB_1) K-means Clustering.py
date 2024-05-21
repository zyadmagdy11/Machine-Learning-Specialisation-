print ("=======================================================")
print ("-----------------------------------------------")

"""
In this exercise, you will implement the K-means algorithm and use it for image compression.
You will start with a sample dataset that will help you gain an intuition of how the K-means algorithm works.
After that, you will use the K-means algorithm for image compression by reducing the number of colors
that occur in an image to only those that are most common in that image.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils1 import *
import math


def find_closest_centroids(X, centroids):
 
    K = centroids.shape[0]

    idx = np.zeros(X.shape[0], dtype=int)
    C_i = np.zeros(K, dtype=float)

    for i in range (X.shape[0]):
        for j in range(K):
           C_i[j] =  np.linalg.norm(X[i] - centroids[j] )     # sqrt((X_x - centroids_x)^2 + (X_y - centroids_y)^2 )
        idx[i] = np.argmin(C_i)

    return idx


def compute_centroids(X, idx, K):
  
    m, n = X.shape
    
    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i] =  ( np.sum(X[idx==i] , axis = 0) )/len(X[idx==i])
        
    return centroids
















X = load_data()
K = 3

initial_centroids = np.array([[3,3], [6,2], [8,5] ])

idx = find_closest_centroids(X, initial_centroids)

centroids = compute_centroids(X, idx, K)


















def run_kMeans(X, centroids, find_closest_centroids , max_iters=10, plot_progress=False):

    def plot_kmeans(X,centroids , previous_centroids , idx):
   
        color = ['r' , 'g' , 'b' , 'c' , 'm' , 'y' , 'k' , 'w' , 'mediumpurple' , 'navy' , 'linen' , 'gainsboro' , 'forestgreen' , 'darkslategray' , 'lightblue' , 'dimgrey' , 'r' , 'g' , 'b' , 'c' , 'm' , 'y' , 'k' , 'w' , 'mediumpurple' , 'navy' , 'linen' , 'gainsboro' , 'forestgreen' , 'darkslategray'  ]
        K = centroids.shape[0]
        fig , ax = plt.subplots (1,1,figsize = (20,6))

        for i in range (K):
            ax.scatter (X[idx == i , 0] ,X[idx == i , 1] , marker='o',c=color[i] , linewidths=0.7 ,facecolor='none')
            ax.scatter (centroids[i,0] , centroids[i,1]  , marker='x', c='m' , s=80 )
            ax.scatter (previous_centroids[i,0] , previous_centroids[i,1]  , marker='x', c='k' , s=80)
            ax.plot ([centroids[i,0],previous_centroids[i,0] ]  , [centroids[i,1],previous_centroids[i,1] ] , color = 'k' , lw = 0.5)

        # plt.legend(loc = 'lower right')
        plt.show()
    
    idx = np.zeros(X.shape[0])
    K = centroids.shape[0]
    previous_centroids = centroids
    plot_kmeans (X,centroids , previous_centroids , idx)


    for i in range (max_iters):
        idx = find_closest_centroids (X, centroids)
        centroids = compute_centroids(X, idx, K)
        plot_kmeans (X,centroids , previous_centroids , idx)
        previous_centroids = centroids

    plt.close('all')
    return centroids, idx


run_kMeans(X , initial_centroids , find_closest_centroids)




"""
Random initialization
"""


def kMeans_init_centroids(X, K):

    
    randidx = np.random.permutation(X.shape[0])
    
    centroids = X[randidx[:K]]
    
    return centroids


K = 3
max_iters = 10
initial_centroids = kMeans_init_centroids(X, K)
centroids, idx = run_kMeans(X, initial_centroids,find_closest_centroids , max_iters, plot_progress=True)


"""
Image compression with K-means

In a straightforward 24-bit color representation of an image 2
each pixel is represented as three 8-bit unsigned integers (ranging from 0 to 255)
that specify the red, green and blue intensity values.
This encoding is often refered to as the RGB encoding.

Our image contains thousands of colors, and in this part of the exercise,
you will reduce the number of colors to 16 colors.

By making this reduction,
it is possible to represent (compress) the photo in an efficient way.

Specifically, you only need to store the RGB values of the 16 selected colors,
and for each pixel in the image you now need to only store the index of the color
at that location (where only 4 bits are necessary to represent 16 possibilities).

In this part, you will use the K-means algorithm to select the 16 colors that will be used to represent the compressed image.

Concretely, you will treat every pixel in the original image
as a data example and use the K-means algorithm
to find the 16 colors that best group (cluster) the pixels in the 3- dimensional RGB space.

Once you have computed the cluster centroids on the image, you will then use the 16 colors to replace the pixels in the original image
"""


original_img = plt.imread('/Users/magdyroshdy/Desktop/Python/Unsupervised Machine Learning/LAB_1) Data/bird_small.png')

plt.imshow(original_img)
plt.show()


""" For example, original_img[50, 33, 2] gives the blue intensity of the pixel at row 50 and column 33."""

print("Shape of original_img is:", original_img.shape)

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))


K = 16
max_iters = 10

initial_centroids = kMeans_init_centroids(X_img, K)
centroids, idx = run_kMeans(X_img, initial_centroids, find_closest_centroids , max_iters)

plot_kMeans_RGB(X_img, centroids, idx, K)
show_centroid_colors(centroids)


X_recovered = centroids[idx, :] 
X_recovered = np.reshape(X_recovered, original_img.shape) 


fig, ax = plt.subplots(1,2, figsize=(12,8))
plt.axis('off')
ax[0].imshow(original_img)
ax[0].set_title('Original')
ax[0].set_axis_off()
ax[1].imshow(X_recovered)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
print ("=======================================================")
print ("-----------------------------------------------")

