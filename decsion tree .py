# -*- coding: utf-8 -*-
"""
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer




Breastmodel= load_breast_cancer()
X = Breastmodel.data
y = Breastmodel.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

DecisionTreeclassiferModel = DecisionTreeClassifier( max_depth=3,random_state=33)
DecisionTreeclassiferModel.fit(X_train, y_train)


print('DecisionTreeRegressor Train Score is : ' ,DecisionTreeclassiferModel.score(X_train, y_train))
print('DecisionTreeRegressor Test Score is : ' , DecisionTreeclassiferModel.score(X_test, y_test))

y_pred = DecisionTreeclassiferModel.predict(X_test)


print("Accuracy:",accuracy_score(y_test, y_pred))


##Accuracy: 0.9202127659574468