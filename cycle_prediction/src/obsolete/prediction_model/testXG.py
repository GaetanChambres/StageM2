#!/usr/bin/env python3

import pandas as pd
from numpy import loadtxt,append
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm


import warnings
warnings.filterwarnings("ignore")

##############################
#####     Open Files     #####
##############################
DS1 = "./data/csv/mfccs_csv/DS1globalfeatures.csv"
DS2 = './data/csv/mfccs_csv/DS2globalfeatures.csv'
DS3 = './data/csv/mfccs_csv/DS3globalfeatures.csv'

# train1_dataset = loadtxt(DS1, delimiter=",")
# train2_dataset = loadtxt(DS2, delimiter=",")
# test_dataset = loadtxt(DS3, delimiter=",")
# print("trained on DS1 & DS2 then tested on DS3")

train1_dataset = loadtxt("./data/csv/challenge/mfcc_train.csv", delimiter=",")
test_dataset = loadtxt("./data/csv/challenge/mfcc_test.csv", delimiter=",")
print("trained then tested according the challenge database")

# train1_dataset = loadtxt(DS2, delimiter=",")
# train2_dataset = loadtxt(DS3, delimiter=",")
# test_dataset = loadtxt(DS1, delimiter=",")
# print("trained on DS2 & DS3 then tested on DS1")

# train1_dataset = loadtxt(DS1, delimiter=",")
# train2_dataset = loadtxt(DS3, delimiter=",")
# test_dataset = loadtxt(DS2, delimiter=",")
# print("trained on DS1 & DS3 then tested on DS2")

# combined_dataset = append(train1_dataset,train2_dataset,axis=0)
combined_dataset = train1_dataset
# print("combined shape")
# print(combined_dataset.shape)

# with open(test_dataset, 'r') as f:
#     for i, l in enumerate(f):
#         pass
#         tmp = i + 1
#     nb_lines = tmp
nb_lines = len(test_dataset)
print(nb_lines)
# print(dataset.shape)
info = combined_dataset[:,:3]
normal_train = combined_dataset[:,3:4]
crackles_train = combined_dataset[:,4:5]
wheezles_train = combined_dataset[:,5:6]
both_train = combined_dataset[:,6:7]
features_train = combined_dataset[:,7:]

normal_stats = 0
crackles_stats = 0
wheezles_stats = 0
both_stats = 0
normal_test = test_dataset[:,3:4]
crackles_test = test_dataset[:,4:5]
wheezles_test = test_dataset[:,5:6]
both_test = test_dataset[:,6:7]
features_test = test_dataset[:,7:]

for n in range(0,nb_lines):
    if normal_test[n] == 1:
        normal_stats += 1
    if crackles_test[n] == 1:
        crackles_stats += 1
    if wheezles_test[n]:
        wheezles_stats += 1
    if both_test[n]:
        both_stats += 1
print(normal_stats)
print(crackles_stats)
print(wheezles_stats)
print(both_stats)
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
#####    Working on normal  #####
#--------------------------------
print('*********************')
print("Working on normal")
model.fit(features_train, normal_train)

# print(model)
# print()
# print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = sklm.accuracy_score(normal_test, predictions)
accuracy_nb = sklm.accuracy_score(normal_test, predictions,normalize=False)
tn, fp, fn, tp = sklm.confusion_matrix(normal_test, predictions).ravel()
sensitivity = tp/(tp+fn)
print("sensitivity : %.2f" % sensitivity)
specificity = tn/(tn+fp)
print("specificity : %.2f" % specificity)
print("nb_preds: %d over %d" % (accuracy_nb,normal_stats))
res = (sensitivity + specificity / 2)
print("accuracy : %.2f%%" % (res))
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')

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
accuracy = sklm.accuracy_score(crackles_test, predictions)
accuracy_nb = sklm.accuracy_score(crackles_test, predictions,normalize=False)

tn, fp, fn, tp = sklm.confusion_matrix(crackles_test, predictions).ravel()
sensitivity = tp/(tp+fn)
print("sensitivity : %.2f" % sensitivity)
specificity = tn/(tn+fp)
print("specificity : %.2f" % specificity)
print("nb_preds : %d over %d" % (accuracy_nb,crackles_stats))
res = (sensitivity + specificity / 2)
print("accuracy : %.2f%%" % (res))
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')
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
accuracy = sklm.accuracy_score(wheezles_test, predictions)
accuracy_nb = sklm.accuracy_score(wheezles_test, predictions,normalize=False)
tn, fp, fn, tp = sklm.confusion_matrix(wheezles_test, predictions).ravel()
sensitivity = tp/(tp+fn)
print("sensitivity : %.2f" % sensitivity)
specificity = tn/(tn+fp)
print("specificity : %.2f" % specificity)
print("nb_preds : %d over %d" % (accuracy_nb,wheezles_stats))
res = (sensitivity + specificity / 2)
print("accuracy : %.2f%%" % (res))
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')

#--------------------------------
#####     Working on both   #####
#--------------------------------
print("Working on both crackles and wheezles")
model.fit(features_train, both_train)

# print(model)
# print()
# print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = sklm.accuracy_score(both_test, predictions)
accuracy_nb = sklm.accuracy_score(both_test, predictions,normalize=False)

tn, fp, fn, tp = sklm.confusion_matrix(both_test, predictions).ravel()
sensitivity = tp/(tp+fn)
print("sensitivity : %.2f" % sensitivity)
# accuracy_nb = accuracy_score(both_test, predictions,normalize=False)
specificity = tn/(tn+fp)
print("specificity : %.2f" % specificity)
print("nb_preds : %d over %d" % (accuracy_nb,both_stats))
res = (sensitivity + specificity / 2)
print("accuracy : %.2f%%" % (res))
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')
