
print ("=======================================================")
print ("-----------------------------------------------")
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility


df = pd.read_csv("/Users/magdyroshdy/Desktop/Python/Advanced Learning Algorithms/heart.csv")
df.head()


cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]


df = pd.get_dummies(data = df, prefix = cat_variables,columns = cat_variables)

df.head()

features = [x for x in df.columns if x not in 'HeartDisease']        ## 20 features


X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)

# min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]         ## If the number is an integer, then it is the actual quantity of samples,
# max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None]                   # None means that there is no depth limit.





# accuracy_list_train = []
# accuracy_list_val = []
# for min_samples_split in min_samples_split_list:
#     # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
#     model = DecisionTreeClassifier(min_samples_split = min_samples_split,
#                                    random_state = RANDOM_STATE).fit(X_train,y_train) 
#     predictions_train = model.predict(X_train) ## The predicted values for the train dataset
#     predictions_val = model.predict(X_val) ## The predicted values for the test dataset
#     accuracy_train = accuracy_score(predictions_train,y_train)
#     accuracy_val = accuracy_score(predictions_val,y_val)
#     accuracy_list_train.append(accuracy_train)
#     accuracy_list_val.append(accuracy_val)

# plt.title('Train x Validation metrics')
# plt.xlabel('min_samples_split')
# plt.ylabel('accuracy')
# plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
# plt.plot(accuracy_list_train)
# plt.plot(accuracy_list_val)
# plt.legend(['Train','Validation'])

# plt.show()

# """
# Note how increasing the the number of min_samples_split reduces overfitting.
# Increasing min_samples_split from 10 to 30, and from 30 to 50,
# even though it does not improve the validation accuracy,
# it brings the training accuracy closer to it,
# showing a reduction in overfitting.
# Let's do the same experiment with max_depth.
# """

# accuracy_list_train = []
# accuracy_list_val = []
# for max_depth in max_depth_list:
#     # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
#     model = DecisionTreeClassifier(max_depth = max_depth,
#                                    random_state = RANDOM_STATE).fit(X_train,y_train) 
#     predictions_train = model.predict(X_train) ## The predicted values for the train dataset
#     predictions_val = model.predict(X_val) ## The predicted values for the test dataset
#     accuracy_train = accuracy_score(predictions_train,y_train)
#     accuracy_val = accuracy_score(predictions_val,y_val)
#     accuracy_list_train.append(accuracy_train)
#     accuracy_list_val.append(accuracy_val)

# plt.title('Train x Validation metrics')
# plt.xlabel('max_depth')
# plt.ylabel('accuracy')
# plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
# plt.plot(accuracy_list_train)
# plt.plot(accuracy_list_val)
# plt.legend(['Train','Validation'])
# plt.show()



# """
# So we can choose the best values for these two hyper-parameters for our model to be:

# max_depth = 4
# min_samples_split = 50
# """

# decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
#                                              max_depth = 3,
#                                              random_state = RANDOM_STATE).fit(X_train,y_train)


# print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
# print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")

# print ("=======================================================")

# """
# Now let's try the Random Forest algorithm also, using the Scikit-learn implementation.

# All of the hyperparameters found in the decision tree model will also exist in this algorithm, since a random forest is an ensemble of many Decision Trees.
# One additional hyperparameter for Random Forest is called n_estimators which is the number of Decision Trees that make up the Random Forest.
# """

# random_forest_model = RandomForestClassifier(n_estimators = 100,
#                                              max_depth = 16, 
#                                              min_samples_split = 10).fit(X_train,y_train)

# print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")



# """
# Next is the Gradient Boosting model, called XGBoost.
# The boosting methods train several trees,
# but instead of them being uncorrelated to each other,
# now the trees are fit one after the other in order to minimize the error.

# The model has the same parameters as a decision tree, plus the learning rate.
# The learning rate is the size of the step on the Gradient Descent method that the XGBoost uses internally to minimize the error on each train step.
# """

n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval

X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)






print ("-----------------------------------------------")
print ("=======================================================")
