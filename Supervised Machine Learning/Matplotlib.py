print ("=======================================================")
print ("-----------------------------------------------")

import matplotlib.pyplot as plt
import numpy as np


xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints,'o' )
plt.show()
print ("-----------------------------1")



xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints , linestyle = 'dotted' )
plt.show()
print ("-----------------------------2")


#If we do not specify the points on the x-axis, they will get the default values 0, 1, 2, 3 (etc., depending on the length of the y-points. 

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints)
plt.show()
print ("-----------------------------3")


# Marked Reference =>  https://www.w3schools.com/python/matplotlib_markers.asp

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = '*' , color = 'r')
plt.show()
print ("-----------------------------4")

# Format Strings 


ypoints = np.array([3, 8, 1, 10])

# marker|line|color
plt.plot(ypoints, 'p--m')
plt.show()
print ("-----------------------------5")

#Marker Size

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = '4', ms = 20)
plt.show()
print ("-----------------------------6")


#marker edge color

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'D', ms = 15, mec = 'm' , color = 'k')
plt.show()


print ("-----------------------------7")

# Line Width

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, linewidth = '20.5')
plt.show()

print ("-----------------------------8")



y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])

plt.plot(y1)
plt.plot(y2)

plt.show()


print ("-----------------------------9")



x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title("Sports Watch Data")
plt.grid()
plt.show()

print ("-----------------------------10")



x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title("Sports Watch Data")
plt.grid(axis= 'x')
plt.show()

print ("-----------------------------11")



x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title("Sports Watch Data")
plt.grid(axis= 'y')
plt.show()

print ("-----------------------------12")



x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)

plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.title("Sports Watch Data")
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()

print ("-----------------------------13")



x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 1)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 2)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 3)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 4)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(2, 3, 5)
plt.plot(x,y)

x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(2, 3, 6)
plt.plot(x,y)
plt.suptitle("MY SHOP")
plt.show()

print ("-----------------------------14")
# Matplotlib Scatter

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x, y , color = 'hotpink')

x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x, y, color = '#88c729')

plt.grid()
plt.show()

print ("-----------------------------15")
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

plt.scatter(x, y, s=sizes, alpha=0.5)
plt.colorbar()
plt.show()

print ("-----------------------------16")

x = np.random.randint(100, size=(100))
y = np.random.randint(100, size=(100))
colors = np.random.randint(100, size=(100))
sizes = 10 * np.random.randint(100, size=(100))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')

plt.colorbar()

plt.show()


print ("-----------------------------16")


x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])

plt.bar(x,y)
plt.show()

plt.show()

print ("-----------------------------17")


x = np.random.normal(170, 10, 250)
y = np.random.normal(170, 10, 250)

plt.scatter(x,y)
plt.show()


print ("-----------------------------18")
y = np.array([35, 25, 25, 15])

plt.pie(y)
plt.show() 
print ("-----------------------------19")


y = np.array([35, 25, 25, 15])
mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

plt.pie(y, labels = mylabels)
plt.show()

print ("=======================================================")
print ("-----------------------------------------------")
