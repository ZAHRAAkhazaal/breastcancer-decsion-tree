# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:22:12 2022

@author: asus
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

logisticregessiondata=load_breast_cancer() 
X=logisticregessiondata.data
y=logisticregessiondata.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=44, shuffle =True)

LogisticRegressionModel = LogisticRegression( penalty='l2',max_iter= 1000, solver='lbfgs',C=.01,random_state=10)
LogisticRegressionModel.fit(X_train, y_train)

print('scoretrain',LogisticRegressionModel.score(X_train,y_train))
print('socretest',LogisticRegressionModel.score(X_test,y_test))

y_pred=LogisticRegressionModel.predict(X_test)

print('accurcy',accuracy_score(y_pred,y_test))
accurcy 0.9574468085106383