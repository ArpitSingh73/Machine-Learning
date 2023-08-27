# The dataseet used, comes from the UCI Machine Learning Repository. This data was collected 
# in 1978. However data is not currently available in Scikit learn, so I fetched the data
# from below mentioned website.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# This should work if data is present in Scikit learn, if not then follow method(2)

#                        Method(1)
# data = sklearn.datasets.load_boston()
# df = pd.DataFrame(data, columns = data.feature_names)
# print(df)

#                       Method(2) 
# fetching the data from given link--->
# 'raw_df' contains data having input and output columns-->
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# Now we will seperate input and output columns in 'data' & 'target' variables-->
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# We have to convert fetched data into Pandas Dataframe-->
data = pd.DataFrame(data ,columns=['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM','AGE','DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT' ])
target = pd.DataFrame(target , columns = ['price'] )

# Adding 'price' column to input data-->
data['price'] = target

# Seperating input and output dataframes-->
X = data.drop(['price'], axis=1)
Y = data['price']

# Spilliting the data into tranning and test datasets-->
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# Creating object of 'XGBRegressor' and 'LinearRegression'-->
xreg = XGBRegressor()
lr = LinearRegression()

# Passing the datasets and trainning the modles-->
xreg.fit(X_train, Y_train)
lr.fit(X_train, Y_train)

# Passing the test datasets and storing the predction-->
prediction1 = xreg.predict(X_test)
prediction2 = lr.predict(X_test)


# Finally checking the 'r2_score' of both the modles-->
# Out of various possible values of 'random_state', for value 3 the r2_score of 'XGBRegressor' model comes to be the best-->
print(metrics.r2_score( Y_test , prediction1))
print(metrics.r2_score( Y_test ,prediction2))