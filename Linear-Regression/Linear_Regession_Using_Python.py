# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 13:28:50 2019

@author: vamsi
"""

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("D:/Bdat_Course_Material/Semister 2/Business Intelligence/Data Sets/50-Startups.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LE = LabelEncoder()
X[:,3] = LE.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)

colors = ("red","blue")
plt.scatter(y_pred,y_test,c=colors)
plt.xlabel("y_pred")
plt.xlabel("y_test")
plt.show()