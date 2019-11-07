import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


dataset = pd.read_csv(r'D:\Bdat_Course_Material\Semister 2\Business Intelligence\Assignment1\voice.csv')





X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 


from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LabelEncoder = LabelEncoder()
y = LabelEncoder.fit_transform


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

corr = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar = True, square = True, cmap = 'coolwarm')
plt.show()

##REMOVING CORELATED VARIABLES


dataset = pd.read_csv(r'D:\Bdat_Course_Material\Semister 2\Business Intelligence\Assignment1\voice.csv')
dataset.drop("skew", axis = 1, inplace = True)
dataset.drop("maxdom", axis = 1, inplace = True)
dataset.drop("centroid", axis = 1, inplace = True)
dataset.drop("sd", axis = 1, inplace = True)
dataset.drop("meanfreq", axis = 1, inplace = True)
dataset.drop("IQR", axis = 1, inplace = True)



X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 


from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

LabelEncoder = LabelEncoder()
y = LabelEncoder.fit_transform


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

corr = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cbar = True, square = True, cmap = 'coolwarm')
plt.show()
