import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Advertising.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
x = df.drop(columns=['sales'])
x.drop(columns=['newspaper'], inplace=True)
y = df.sales


print(x.head)

linear = LinearRegression()

linear.fit(x, y)

print(linear.coef_)

print(linear.intercept_)

print(linear.predict([[230.1, 37.8]]))

pickle.dump(linear, open('linear_model.pickle', 'wb'))
