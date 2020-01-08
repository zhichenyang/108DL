# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:10:51 2019

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder 



#把資料集做preprocessing 變成dataframe
df = pd.read_csv("abc.csv")
#tmp = df.keys()[8]
df1 = df.iloc[:,0:8]

print(df1)

# One Hot Encodes 
one_hot_cols = df1.columns.tolist()

#one_hot_cols.remove('salary')
dataset_bin_enc = pd.get_dummies(df1, columns=one_hot_cols)
#print(dataset_bin_enc.head())
#dataset_bin_enc.head()
#print(type(df.iloc[:,[8]]))
#encoder = OneHotEncoder(sparse=False)
#target_salary = encoder.fit_transform(df.iloc[:,[8]])

df['salary'] = df['salary'].map({'<=50K':1,'>50K':0}).astype(int)
#print(df['salary'])
#df.keys()[8]
#df.info()
#df_feat.head(6)
#print(type(df))
#print(df.DESC)

from sklearn.model_selection import train_test_split

X = dataset_bin_enc
y=df['salary']
print(X)
#將資料分成訓練集和測試集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=16,stratify=y)

#print(X_train)
#載入support vector classifier套件
from sklearn.svm import SVC
model = SVC()

#使用support vector classifier建立模型
model.fit(X_train,y_train)

#利用測試組資料測試模型結果
prediction = model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n")
print(confusion_matrix(y_test,prediction))
print('\n')
print("Classification report:\n")
print(classification_report(y_test,prediction))

from sklearn import metrics

#印出accuracy
accuracy = metrics.accuracy_score(y_test,prediction)
print("Accuracy: ",accuracy)

#印出precision
precision = metrics.precision_score(y_test,prediction,pos_label=3,average=None)
print("Precision: ",precision)

#印出recall
recall = metrics.recall_score(y_test,prediction,pos_label=3,average=None)
print("Recall:",recall)


fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction, pos_label=2)
print("AUC: ",metrics.auc(fpr, tpr))
print("\n")
#AUC = metrics.roc_auc_score(y_test,prediction,average=None)
"""from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)

grid.fit(X_train,y_train)
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))"""
