# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:37:46 2019

@author: vamsi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

## Import the iris dataset
iris = datasets.load_iris()
## Create data frame using features of iris data
X = pd.DataFrame(iris.data)
# Assign column names
X.columns = ['sepal_length','sepal_width','petal_length','petal_width']
# Create a data frame for target variable
y= pd.DataFrame(iris.target)
y.columns = ['Targets']
# We use KMeans clustering to cluster our data to 3 groups
model = KMeans(n_clusters=3)
model.fit(X)
print(model.labels_)
# Visualize  cluster
colormap = np.array(['red','blue','green'])
plt.scatter(X.petal_length,X.petal_width,c=colormap[model.labels_],s=40)
# Comapre Target vs Clusters
plt.title("KMeans cluster")
plt.show()
# Plot our target value
plt.scatter(X.petal_length,X.petal_width,c=colormap[y.Targets],s=40)
plt.title("Actual cluster")
plt.show()