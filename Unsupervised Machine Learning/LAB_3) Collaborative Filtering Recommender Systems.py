print ("=======================================================")
print ("-----------------------------------------------")

import numpy as np
import tensorflow as tf
from tensorflow import keras , linalg
from recsys_utils import *



#Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

print("Y", Y.shape, "R", R.shape)
print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("num_features", num_features)
print("num_movies",   num_movies)
print("num_users",    num_users)

tsmean =  np.mean(Y[0, R[0, :].astype(bool)])
print(f"Average rating for movie 1 : {tsmean:0.3f} / 5" )

num_users_r = 4
num_movies_r = 5 
num_features_r = 3

X_r = X[:num_movies_r, :num_features_r]
W_r = W[:num_users_r,  :num_features_r]
b_r = b[0, :num_users_r].reshape(1,-1)
Y_r = Y[:num_movies_r, :num_users_r]
R_r = R[:num_movies_r, :num_users_r]

def cofi_cost_func(X, W, b, Y, R, lambda_):

    nm, nu = Y.shape
    J = 0
    cost_term = 0
    reg_term = 0
    ### START CODE HERE ###  
    for j in range (nu):
        w = W[j,:]
        b_j = b[0,j]
        for i in range (nm):
            cost_term += R[i,j] * np.square( np.dot (w , X[i] ) + b_j - Y[i,j] ) 
            
    reg_term += (lambda_/2) * (np.sum(X**2) + np.sum (W**2))
    J = 0.5 * cost_term + reg_term
    return J


def cofi_cost_func_v(X, W, b, Y, R, lambda_):
  
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J




J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 0)
print(f"Cost: {J:0.2f}")

J = cofi_cost_func_v(X_r, W_r, b_r, Y_r, R_r, 1.5)
print(f"Cost (with regularization): {J:0.2f}")



movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies)          #  Initialize my ratings

# Check the file small_movie_list.csv for id of each movie in our dataset
# For example, Toy Story 3 (2010) has ID 2700, so to rate it "5", you can set
my_ratings[2700] = 5 

#Or suppose you did not enjoy Persuasion (2007), you can set
my_ratings[2609] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[929]  = 5   # Lord of the Rings: The Return of the King, The
my_ratings[246]  = 5   # Shrek (2001)
my_ratings[2716] = 3   # Inception
my_ratings[1150] = 5   # Incredibles, The (2004)
my_ratings[382]  = 2   # Amelie (Fabuleux destin d'Amélie Poulain, Le)
my_ratings[366]  = 5   # Harry Potter and the Sorcerer's Stone (a.k.a. Harry Potter and the Philosopher's Stone) (2001)
my_ratings[622]  = 5   # Harry Potter and the Chamber of Secrets (2002)
my_ratings[988]  = 3   # Eternal Sunshine of the Spotless Mind (2004)
my_ratings[2925] = 1   # Louis Theroux: Law & Disorder (2008)
my_ratings[2937] = 1   # Nothing to Declare (Rien à déclarer)
my_ratings[793]  = 5   # Pirates of the Caribbean: The Curse of the Black Pearl (2003)
my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');


Y, R = load_ratings_small()

Y = np.c_[my_ratings, Y]

R = np.c_[(my_ratings != 0).astype(int), R]

Ynorm, Ymean = normalizeRatings(Y, R)







print ("=======================================================")
print ("-----------------------------------------------")

