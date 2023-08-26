# This is my own class for model training
# This class shows the maths involved behind the scene
# or should say the core Algorithm.

class CustomLR():
    def __init__(self):
        self.slope = None
        self.interCept = None

    def fit(self, X_train, y_train):
        numerator = 0
        denominator = 0

        for i in range(X_train.shape[0]):
            numerator += (X_train[i] - X_train.mean())*(y_train[i] - y_train.mean())
            denominator += (X_train[i] - X_train.mean())*(X_train[i] - X_train.mean())

        self.slope = numerator/denominator
        self.interCept = y_train.mean() - (self.slope*X_train.mean())

    def predict(self, X_test):
       return self.interCept + self.slope*X_test



import pandas as pd
import numpy as np

# Use your own path for data
df = pd.read_csv('Machine Learning\Machine-Learning\Simple Linear regression\Salary.csv')

# print(df.head())


# x = df.iloc[:,0]
# y = df.iloc[:,1]

x = df.iloc[:,0:1]
y = df.iloc[:,-1]

# print(x, y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Object of Scikit Learn and Object of my own class
# both can be used. 
lr = LinearRegression()

lr.fit(x_train.values, y_train.values)

# Code for visualizing the regrfession line generated by algorithm
plt.scatter(df['YearsExperience'], df['Salary'])
plt.plot(x_test, lr.predict(x_test.values), color= 'blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
# print(plt.show())

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# However for accuracy measure my class has no option for that, so better
# to go for Scikit Learn's metrics
print(r2_score(y_test, y_pred))
# print(y_pred)

# x = np.array([10]).reshape(1,1)
# print("Predicted salary is --> ", lr.predict(x))
# code for object of Scikit Learn 
# print("Value of intercept is --> ",lr.intercept_)
# print("Value of slope is --> ",lr.coef_)


# code for object of my own class
# print("Value of intercept is --> ",lr.interCept)
# print("Value of slope is --> ",lr.slope)
