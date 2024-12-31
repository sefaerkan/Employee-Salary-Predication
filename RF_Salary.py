import pandas as pd
import numpy as np

data = pd.read_csv('employee_data.csv')
data.head()
data.info()

data.describe(include='all')

position = data['Position'].value_counts()
print("\nNumber of employees by position")
print(position)

gender_counts = data['Gender'].value_counts()
print("\nGender Distribution")
print(gender_counts)

from sklearn.preprocessing import LabelEncoder
LabelEncoder = LabelEncoder()
data['Position'] = LabelEncoder.fit_transform(data['Position'])
data['Gender'] = LabelEncoder.fit_transform(data['Gender'])

data.head()

X = data.iloc[:, 1:4]
X

y = data.iloc[:, -1]
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X_train), len(X_test), len(y_train), len(y_test)

from sklearn.ensemble import RandomForestRegressor
Model = RandomForestRegressor(n_estimators=100, random_state=1)
Model.fit(X, y)

y_pred = Model.predict(X_test)

Model.score(X_test, y_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mse

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual Salary vs Predicted Salary')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

