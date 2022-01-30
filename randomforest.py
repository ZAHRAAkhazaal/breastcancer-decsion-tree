# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 20:13:38 2022

@author: asus
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

Breastcancerdata=load_breast_cancer() 
X=Breastcancerdata.data
y=Breastcancerdata.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33, random_state=44, shuffle =True)

RandomForestclassifierModel = RandomForestClassifier(n_estimators=100,max_depth=2, random_state=33)
RandomForestclassifierModel.fit(X_train, y_train)

print('scoretrain',RandomForestclassifierModel.score(X_train,y_train))
print('socretest',RandomForestclassifierModel.score(X_test,y_test))

y_pred=RandomForestclassifierModel.predict(X_test)

print('accurcy',accuracy_score(y_pred,y_test))
##accurcy 0.9414893617021277