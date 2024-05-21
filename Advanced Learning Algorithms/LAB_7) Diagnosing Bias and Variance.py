
print ("=======================================================")
print ("-----------------------------------------------")
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import utils1


############################ Fixing High Bias ############################
##########################################################################


# x_train with one feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils1.prepare_dataset('/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/LAB_7) Diagnosing Bias and Variance/c2w3_lab2_data1.csv')

model = LinearRegression()

# Train and plot polynomial regression models
# utils1.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=400)



"""
As you can see, the more polynomial features you add,
the better the model fits to the training data.
In this example, it even performed better than the baseline.
At this point, you can say that the models with degree greater than 4 are low-bias
because they perform close to or better than the baseline.

However, if the baseline is defined lower
(e.g. you consulted an expert regarding the acceptable error),
then the models are still considered high bias.
You can then try other methods to improve this.
"""

# x_train with Two feature
# utils1.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=250)



"""
Try getting additional features
Another thing you can try is to acquire other features.
Let's say that after you got the results above,
you decided to launch another data collection campaign that captures another feature.
Your dataset will now have 2 columns for the input features as shown below.
"""

x_train, y_train, x_cv, y_cv, x_test, y_test = utils1.prepare_dataset('/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/LAB_7) Diagnosing Bias and Variance/c2w3_lab2_data2.csv')


model = LinearRegression()

# Train and plot polynomial regression models. Dataset used has two features.
# utils1.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)




"""
At this point, you might want to introduce regularization to avoid overfitting.
One thing to watch out for is you might make your models underfit if you set the regularization parameter too high.
"""


# Define lambdas to plot
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]

# Define degree of polynomial and train for each value of lambda
# utils1.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

"""
The resulting plot shows an initial  𝜆 of 10 and as you can see,
the training error is worse than the baseline at that point.
This implies that it is placing a huge penalty on the w parameters
and this prevents the model from learning more complex patterns in your data.
As you decrease  𝜆,
the model loosens this restriction and the training error is able to approach the baseline performance.
"""


############################ Fixing High Variance ########################
##########################################################################


# Define lambdas to plot
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# Define degree of polynomial and train for each value of lambda
utils1.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

"""
in contrast to the last exercise above,
setting a very small value of the regularization parameter will keep the model low bias
but might not do much to improve the variance.
As shown below, you can improve your cross validation error by increasing the value of  𝜆.
"""


"""
To illustrate how removing features can improve performance,
you will do polynomial regression for 2 datasets:
the same data you used above (2 features) and another with a random ID column (3 features).
You can preview these using the cell below.
Notice that 2 columns are identical and a 3rd one is added to include random numbers.
"""




# Prepare dataset with randomID feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils1.prepare_dataset('/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/LAB_7) Diagnosing Bias and Variance/c2w3_lab2_data3.csv')



"""
Now you will train the models and plot the results.
The solid lines in the plot show the errors for the data with 2 features.
The dotted lines show the errors for the dataset with 3 features.
As you can see, the one with 3 features has higher cross validation error
Especially as you introduce more polynomial terms.
This is because the model is also trying to learn from the
random IDs even though it has nothing to do with the target.

Another way to look at it is to observe the points at degree=4.
You'll notice that even though the training error is lower with 3 features,
The gap between the training error and cross validation error is a lot wider than when you only use 2 features.
This should also warn you that the model is overfitting.
"""

# Define the model
model = LinearRegression()

# Define properties of the 2 datasets
file1 = {'filename':'/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/LAB_7) Diagnosing Bias and Variance/c2w3_lab2_data3.csv', 'label': '3 features', 'linestyle': 'dotted'}
file2 = {'filename':'/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/LAB_7) Diagnosing Bias and Variance/c2w3_lab2_data2.csv', 'label': '2 features', 'linestyle': 'solid'}
files = [file1, file2]

# Train and plot for each dataset
utils1.train_plot_diff_datasets(model, files, max_degree=4, baseline=250)



################## Get more training examples ##################
################################################################

"""
Lastly, you can try to minimize the cross validation error by getting more examples.
In the cell below, you will train a 4th degree polynomial model
then plot the learning curve of your model
to see how the errors behave when you get more examples.
"""



x_train, y_train, x_cv, y_cv, x_test, y_test = utils1.prepare_dataset('/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/LAB_7) Diagnosing Bias and Variance/c2w3_lab2_data4.csv')

# Instantiate the model class
model = LinearRegression()

utils1.train_plot_learning_curve(model, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

"""
From the results, it shows that the cross validation error starts
to approach the training error as you increase the dataset size.
Another insight you can get from this is that adding more examples
will not likely solve a high bias problem.
That's because the training error remains relatively flat even as the dataset increases.
"""


# Wrap Up

"""
In this lab, you were able to practice how to address high bias and high variance in your learning algorithm.
By learning how to spot these issues,
you have honed your intuition on what to try next when developing your machine learning models.
In the next lectures, you will look deeper into the machine learning development process
and explore more aspects that you need to take into account when working on your projects.
"""



print ("-----------------------------------------------")
print ("=======================================================")
