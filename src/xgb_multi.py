#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

train_dataset = np.loadtxt("./data/csv/challenge/train_tmp_features.csv", delimiter=",")
test_dataset = np.loadtxt("./data/csv/challenge/test_tmp_features.csv", delimiter=",")

# header = train_dataset[:1,:]
classification_train = train_dataset[1:,3:4]
features_train = train_dataset[1:,4:]

classification_test = test_dataset[:,3:4]
features_test = test_dataset[:,4:]

nb_lines = len(test_dataset)
normal = 0
crackles = 0
wheezles = 0
both = 0
for n in range(0,nb_lines):
    if classification_test[n] == 1:
        normal += 1
    if classification_test[n] == 2:
        crackles += 1
    if classification_test[n] == 3:
        wheezles += 1
    if classification_test[n] == 4:
        both += 1

model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,class_num=4,
       silent=True, subsample=1)

#--------------------------------
#####    MULTICLASS MODEL   #####
#--------------------------------
print("MULTICLASS MODEL")
model.fit(features_train, classification_train)

# print(model)
# print()
# print(model.feature_importances_)

# make predictions for test data
y_pred = model.predict(features_test)
predictions = [round(value) for value in y_pred]

cc = tc = cw = tw = cb = tb = cn = tn = 0
for i in range(0,len(classification_test)):
    vr = classification_test[i]
    vp = predictions[i]
    if vr == 2:
        tc+=1
        if vp == 2:
            cc+=1
    if vr == 3:
        tw+=1
        if vp == 3:
            cw+=1
    if vr == 4:
        tb+=1
        if vp == 4:
            cb+=1
    if vr == 1:
        tn+=1
        if vp == 1:
            cn+=1
print("cc = %d, tc = %d, cw = %d, tw = %d, cb = %d, tb = %d, cn = %d, tn = %d" % (cc,tc,cw,tw,cb,tb,cn,tn))
# evaluate predictions
accuracy = sklm.accuracy_score(classification_test, predictions)
accuracy_nb = sklm.accuracy_score(classification_test, predictions,normalize=False)

# ***************************************************************
# TODO : revoir le calcul ce Se et Sp en fonction DES 4 CLASSES !
# ***************************************************************
tn1, fp1, fn1, tp1, tn2, fp2, fn2, tp2, tn3, fp3, fn3, tp3,tn4, fp4, fn4, tp4 = sklm.confusion_matrix(classification_test, predictions).ravel()

sensitivity_num = cc+cw+cb
sensitivity_denum = tc+tw+tb
sensitivity_global = sensitivity_num/sensitivity_denum
specificity_global = cn/tn
accuracy = (sensitivity_global + specificity_global)/2
# print("sensitivity from confusion matrix : %.2f" % sensitivity_confusion)
# print("sensitivity from perso parsing : %.2f" % sensitivity_perso)
print("sensitivity global : %.2f" % sensitivity_global)
print("specificity global : %.2f" % specificity_global)
print("accuracy : %.2f" % accuracy)
# specificity1 = tn1/(tn1+fp1)
# print("specificity1 : %.2f" % specificity1)
# accuracy_nb = sklm.accuracy_score(classification_test, predictions,normalize=False)
# print("nb_preds : %d over %d" % (accuracy_nb,normal))
# res1 = (sensitivity1 + specificity1 / 2)
# print("accuracy : %.2f%%" % (res1))
# # print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')
