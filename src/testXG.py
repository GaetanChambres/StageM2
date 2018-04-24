#!/usr/bin/env python3


import pandas as pd
from numpy import loadtxt,append
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings("ignore")

##############################
#####     Open Files     #####
##############################
DS1 = "./data/csv/mfccs_csv/DS1globalfeatures.csv"
DS2 = './data/csv/mfccs_csv/DS2globalfeatures.csv'
DS3 = './data/csv/mfccs_csv/DS3globalfeatures.csv'

train1_dataset = loadtxt(DS2, delimiter=",")
train2_dataset = loadtxt(DS3, delimiter=",")

test_dataset = loadtxt(DS1, delimiter=",")

print("trained on DS2 & DS3 then tested on DS1")

combined_dataset = append(train1_dataset,train2_dataset,axis=0)
# print("combined shape")
# print(combined_dataset.shape)

# print(dataset.shape)
info = combined_dataset[:,:3]
crackles_train = combined_dataset[:,3:4]
wheezles_train = combined_dataset[:,4:5]
features_train = combined_dataset[:,5:]

crackles_test = test_dataset[:,3:4]
wheezles_test = test_dataset[:,4:5]
features_test = test_dataset[:,5:]

#--------------------------------
#####   Creating the model  #####
#--------------------------------

model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

#--------------------------------
#####   Working on crackles #####
#--------------------------------
print("Working on crackles")
model.fit(features_train, crackles_train)

# print(model)
# print()
# print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(crackles_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

#--------------------------------
#####   Working on wheezles #####
#--------------------------------
print("working on wheezles")
model.fit(features_train, wheezles_train)

# print(model)
# print()
# print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(wheezles_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
