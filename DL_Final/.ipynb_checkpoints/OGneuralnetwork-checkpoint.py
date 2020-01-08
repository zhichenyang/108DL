# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:30:09 2020

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:24:27 2020

@author: User
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.read_adult_data import read_data as read_adult
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

DATA, INTUITIVE_ORDER = read_adult()

df = pd.DataFrame(DATA)

df.columns = [
    "age", "workClass", "education-num", "marital-status", "occupation",
    "race", "gender", "native-country","income"
]

df['income'] = df['income'].map({ "<=50K": 1, ">50K": 0 })
#print(df)

y_all = to_categorical(df['income'].values).astype(int) 
#print(y_all)

df.drop('income', axis=1, inplace=True,)
df = pd.get_dummies(df, columns=[
    "age", "workClass", "education-num", "marital-status", "occupation",
    "race", "gender", "native-country"
])
#print(df.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    df, y_all, test_size=0.2, stratify=y_all,
)


model = Sequential()

model.add(Dense(units=82,input_dim=df.shape[1],kernel_initializer='normal',activation='relu'))

model.add(Dense(units=2,kernel_initializer='normal',activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy']) 

train_history = model.fit(x=X_train, y=y_train, validation_split=0.2, epochs=10, batch_size=100, verbose=2) 

prediction = model.predict(X_test)
print(prediction)

pred= np.array(np.arange(6033))
ytest= np.array(np.arange(6033))

for i,row in enumerate(prediction):
    if prediction[i][0]>prediction[i][1]:
        pred[i]=1
    else:
        pred[i]=0
    
for i,row in enumerate(y_test):
    if y_test[i][0]==1:
        ytest[i]=1
    else:
        ytest[i]=0

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:")
print("\n",confusion_matrix(ytest,pred))
print('\n')
print("Classification report:")
print('\n',classification_report(ytest,pred))

from sklearn import metrics
#印出accuracy
accuracy = metrics.accuracy_score(ytest,pred)
print("Accuracy: ",accuracy)

#印出precision
precision = metrics.precision_score(ytest,pred,pos_label=1,average=None)
print("Precision: ",precision)

#印出recall
recall = metrics.recall_score(ytest,pred,pos_label=1,average=None)
print("Recall:",recall)


fpr, tpr, thresholds = metrics.roc_curve(ytest, pred,pos_label=1)
print(tpr)
print("AUC: ",metrics.auc(fpr, tpr))

if precision[0]>precision[1]:
    maximum = precision[0]
else:
    maximum = precision[1]
missclassification_error = 1-maximum
print("Missclassification error: ",missclassification_error)
