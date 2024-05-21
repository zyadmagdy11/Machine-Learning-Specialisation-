print ("=======================================================")
print ("-----------------------------------------------")


import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(arr)

print(type(arr))

print ("-----------------------1")



arr = np.array(42)

print(arr)


print ("-----------------------2")


arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)

print ("-----------------------3")


a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

print ("-----------------------4")

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

for i in range(len(arr)):
    for j in range(len(arr[0])):
        print (arr[i,j],end = " ")
    print()    


print('2nd element on 1st row: ', arr[0, 1])
print('5th element on 2nd row: ', arr[1, 4])


arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[0, 1, 2])



"""
    Range of values (maximum - minimum) along an axis.

     Examples
    --------
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])

    >>> np.ptp(x, axis=1)
    array([8, 6])

    >>> np.ptp(x, axis=0)
    array([2, 0, 5, 2])

    >>> np.ptp(x)
    10

    This example shows that a negative value can be returned when
    the input is an array of signed integers.

    >>> y = np.array([[1, 127],
    ...               [0, 127],
    ...               [-1, 127],
    ...               [-2, 127]], dtype=np.int8)
    >>> np.ptp(y, axis=1)
    array([ 126,  127, -128, -127], dtype=int8)
    """



print ("-----------------------5")
#NumPy Array Slicing


arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])


arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[:4])

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[::2])

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4])


arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:1, 2])


print ("-----------------------6")
#NumPy Data Types

arr = np.array([1, 2, 3, 4])

print(arr.dtype)

arr = np.array(['apple', 'banana', 'cherry'])

print(arr.dtype)

arr = np.array([1, 2, 3, 4], dtype='S')

print(arr)
print(arr.dtype)

print ("-----------------------7")

arr = np.array([1, 2, 3, 4], dtype='i4')

print(arr)
print(arr.dtype)


arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('U')

print(newarr)
print(newarr.dtype)


arr = np.array([1, 0, 3])

newarr = arr.astype(bool)

print(newarr)
print(newarr.dtype)


print ("-----------------------8")

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)


arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42

print(arr)
print(x)

arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31

print(arr)
print(x)

arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)


print ("-----------------------9")
#NumPy Array Shape

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape)

for i in range (arr.shape[0]):
    for j in range (arr.shape[1]):
        print (arr[i,j] , end = " ")
    print()    



arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)

#NumPy Array Reshaping
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)

print(newarr)

print ("-----------------------10")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)


print ("-----------------------11")
arr = np.array([[[1, 2, 3], [4, 5, 6]] , [[20, 30, 3], [90, 51, 62]]] )

newarr = arr.reshape(-1)

print(newarr)



print ("-----------------------12")

# Iterating Arrays
arr = np.array([1, 2, 3])

for x in arr:
  print(x)


arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)

for x in arr:
  for y in x:
    print(y)


# 3 dimension
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  for y in x:
    for z in y:
      print(z)


print ("-----------------")

arr = np.array([[[13, 21], [32, 4]], [[56, 86], [87, 89]]])

for x in np.nditer(arr):
  print(x)



arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)





print ("-----------------------13")
# NumPy Joining Array


arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))
print (arr)

print ("-----")

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)

print ("-----")

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.hstack((arr1, arr2))

print(arr)

print ("-----")

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.vstack((arr1, arr2))

print(arr)


#NumPy Splitting Array
print ("-----------------------14")

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)


arr = np.array([13, 42, 13, 4, 5, 6])
newarr = np.array_split(arr, 4)
print(newarr)


arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print(newarr)



#NumPy Searching Arrays
print ("----------------------------15")

arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x[0])



arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 0)
print(x)


arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 1)
print(x)


print ("----------------------------16")
#Sorting Arrays

arr = np.array([3, 2, 0, 1])
print(np.sort(arr))

arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))

arr = np.array([True, False, True])
print(np.sort(arr))

arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr))


print ("----------------------------17")
#NumPy Filter Array

arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x]
print(newarr)




arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

print ("-========================================")
##################################################       
        # Random Numbers in NumPy
##################################################       
print ("-========================================")

from numpy import random

x = random.randint(100)                 #Generate a random integer from 0 to 100:
print(x)

x=random.randint(100, size=(5))         #Generate a 1-D array containing 5 random integers from 0 to 100:
print(x)


x = random.randint(100, size=(3, 5))                        #Generate a 2-D array with 3 rows, each row containing 5 random integers from 0 to 100:
print(x)

print ("=============")

x = random.rand()                                            #Generate a random float from 0 to 1:
print(x)

x = random.rand(5)                                          #Generate a 1-D array containing 5 random floats:
print(x)

x = random.rand(3, 5)                                       #Generate a 2-D array with 3 rows, each row containing 5 random numbers:
print(x)

print ("=============")
x = random.choice([3, 5, 7, 9])                             #Return one of the values in an array:
print(x)


x = random.choice([3, 5, 7, 9], size=(3, 5))                #Generate a 2-D array that consists of the values in the array parameter (3, 5, 7, and 9):
print(x)

print ("========================================1")
#Random Data Distribution


x = random.choice([3,5,7,9] , p = [0.1,0.3,0.6,0] , size = (3,5))
print (x)



print ("========================================2")
#Random Permutations and Shuffling

arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr)
print(arr)




arr = np.array([1, 2, 3, 4, 5])
print(random.permutation(arr))
print ("========================================3")

#Visualize Distributions With Seaborn

import matplotlib.pyplot as plt
import seaborn as sns

# sns.distplot([112, 12, 23, 390, 4, 5], hist=False)

# plt.show()               


print ("========================================4")

#Normal Distribution

x = random.normal(size=(2, 3))
print(x)



x = random.normal(loc=1, scale=2, size=(100))         
print(x)                                                              #loc - (Mean) where the peak of the bell exists.
# sns.distplot(x)                                                       #scale - (Standard Deviation) how flat the graph distribution should be.
# plt.show()                                                            #size - The shape of the returned array.


print ("========================================5")
#Binomial Distribution


x = random.binomial(n=10, p=0.5, size=10)
print(x)


# sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
# # plt.show()


print ("========================================6")
#Poisson Distribution

x = random.poisson(lam=2, size=10)
print(x)


# sns.distplot(random.poisson(lam=2, size=1000), kde=False)
# plt.show()


# sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
# sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')
# plt.show()






print ("========================================7")

#Finding LCM (Lowest Common Multiple)
num1 = 9
num2 = 6
x = np.lcm(num1 , num2)
print(x)



arr = np.array([3, 6, 9])
x = np.lcm.reduce(arr)
print(x)





import os
print(os.getcwd())
















arr = np.zeros(4)
arr = input().split()
print(arr)


























































print ("=======================================================")
print ("-----------------------------------------------")
