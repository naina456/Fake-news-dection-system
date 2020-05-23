# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:33:26 2020

@author: Naina
"""

import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
data=pd.read_csv("news.csv",na_values=[" ?"])
#data-preprocessing
print(data.isnull().sum())
#let see in a particular row if anyone of vaalue is missing or both are missing
missing=data[data.isnull().any(axis=1)]
#axis=1--> to consider atleast one missing value in any row
''' Points to note:
1.Missing values in job type-1809
2..Missing values in job type-1816
3.1809 is the occurenece when both columns have nan values
4. 1816-1809=7 there are another 7 rows where value for
occupation is nan ,it is because the jobtype their is NeverWorked
'''
# we don't have mechanism here to deal with  missing values so we will drop them
data2=data.dropna(axis=0)
print("______________________________________________________________________")
print(data2.shape)
#LOGISTIC REGRESSION
#reindexing salary status name to 0,1
data2['label']=data2['label'].map({"FAKE":0,"REAL":1})
print("_______________________________________________________________________")
print(data2['label'])
print("_______________________________________________________________________")
#one hot encoding
new_data=pd.get_dummies(data2,drop_first=True)
#all string values are mapped to integer values

print(new_data)
print("_______________________________________________________________________")
#storing the columns names
columns_list=list(new_data.columns)
print(columns_list)

#separating output varaiable from input variables
print("_______________________________________________________________________")
features=list(set(columns_list)-set(['label']))
#features=features.sort()
print(features)

print("_______________________________________________________________________")
y=data2['label'].values
print(y)
#[0 0 1 ... 0 0 0]
print("_______________________________________________________________________")
x= new_data[features].values
print(x)
#divide data for train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#make an instance of model
logistic =LogisticRegression()
#fittiung the values of x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_
#prediction from test data
prediction=logistic.predict(test_x)
print(prediction)
#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)
#accuracy score
accuracy=accuracy_score(test_y,prediction)
print(accuracy)
#misclassified examples
print("Misclassified values:%d"%(test_y!=prediction).sum())
#dropping a few columns
#knn
 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt
KNNClassifier= KNeighborsClassifier(n_neighbors=17)
KNNClassifier.fit(train_x,train_y)
prediction=KNNClassifier.predict(test_x)
#confusion_matrix=confusion_matrix(test_y,prediction)
accuracy=accuracy_score(test_y,prediction)
print(accuracy)

