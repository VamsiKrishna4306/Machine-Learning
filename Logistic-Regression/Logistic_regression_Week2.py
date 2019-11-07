# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:55:17 2019

@author: vamsi
"""
# import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load our dataset
dataset = pd.read_csv("D:/Bdat_Course_Material/Semister 2/Business Intelligence/Data Sets/Social_Network_Ads.csv")

##  seperate dependent and dependent and independent variables 2nd and ,3rd columns
## last column is the one we predict
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

## Scale our value for better performance 

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

## we scale our independent variable of Train,Test dataset

X_train=sc.fit_transform(X_train)
x_test=sc.fit_transform(x_test)

## Create our logistic Model

from sklearn.linear_model import LogisticRegression

clasifier = LogisticRegression(random_state=0)

## Fit our data to Model

clasifier.fit(X_train,y_train)

y_pred = clasifier.predict(x_test)

## Instead of printing y_pred we will calculate confusion matrix

from sklearn.metrics import confusion_matrix
from sklearn import metrics

cm = confusion_matrix(y_test,y_pred)
print(cm)

print(metrics.accuracy_score(y_pred,y_test))













