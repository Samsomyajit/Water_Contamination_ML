# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:25:07 2019

@author: Dell
"""

import pandas as pd

dataset=pd.read_csv('Water_contamination.csv')

#Dividing Dataset

X= dataset.iloc[:,2:-1].values
Y= dataset.iloc[:,-1].values

#missing values fix

from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean')
imputer=imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)


#predict

Y_pred= classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cn = confusion_matrix(Y_test,Y_pred)

