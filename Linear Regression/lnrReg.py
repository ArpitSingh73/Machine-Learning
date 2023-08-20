import pandas as pd
import numpy as np

df = pd.read_csv('Machine Learning\Linear regression\Salary.csv')

# print(df.head())


# x = df.iloc[:,0]
# y = df.iloc[:,1]

x = df.iloc[:,0:1]
y = df.iloc[:,-1]

# print(x, y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


lr = LinearRegression()
lr.fit(x_train.values, y_train.values)

plt.scatter(df['YearsExperience'], df['Salary'])
plt.plot(x_test, lr.predict(x_test.values), color= 'blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
# print(plt.show())


x = np.array([10]).reshape(1,1)
print("Predicted salary is --> ", lr.predict(x))
print("Value of intercept is --> ",lr.intercept_)

print("Value of slope is --> ",lr.coef_)
