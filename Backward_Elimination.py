#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:13:20 2018

@author: AbhiramTripuraneni
"""
#Multiple linear Regression

#Data Preprocessing

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset= pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 4].values


# As we have cateogorical variables here we change those to numerics
#Encoding cateogorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])

onehotencoder= OneHotEncoder(categorical_features= [3])
X= onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy varibale trap
X=X[:, 1:]


#Split the data set into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

# Fitting multiple Linear Regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Build optimal model using Backward Elimination
import statsmodels.formula.api as sm
#This is for adding a column of 1s which is required for the statsmodel package
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt= X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
#removing the predictor with the highest p-value
X_opt= X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
# This will be the last set of predictors whose p-values are below 0.05 and are significant predictors of dependent vaiable Y.
X_opt= X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()























