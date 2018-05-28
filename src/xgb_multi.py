#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

input_train = "./data/csv/challenge/train_lowlevel.csv"
input_test = "./data/csv/challenge/test_lowlevel.csv"
with open(input_train) as f1:
    line1 = f1.readline()
    data1= line1.split(',')
    print(data1)
    nbcols_train = len(data1)
    print(nbcols_train)
with open(input_test) as f2:
    line2 = f2.readline()
    data2 = line2.split(',')
    print(data2)
    nbcols_test = len(data2)
    print(nbcols_test)



train_dataset = np.loadtxt(input_train, delimiter=",", skiprows = 1, usecols=range(1,nbcols_train))
test_dataset = np.loadtxt(input_test, delimiter=",", skiprows = 1, usecols=range(1,nbcols_test))
# train_dataset = np.loadtxt("./data/csv/challenge/mfcc_train.csv", delimiter=",")
# test_dataset = np.loadtxt("./data/csv/challenge/mfcc_test.csv", delimiter=",")
# header = train_dataset[:1,:]
classification_train = train_dataset[:,0:1]
features_train = train_dataset[:,1:]

classification_test = test_dataset[:,0:1]
features_test = test_dataset[:,1:]

nb_lines = len(test_dataset)
normal = 0
crackles = 0
wheezles = 0
both = 0

for n in range(0,nb_lines):
    if classification_test[n] == 1: # normal
        normal += 1
    if classification_test[n] == 2: # crackle
        crackles += 1
    if classification_test[n] == 3: # wheeze
        wheezles += 1
    if classification_test[n] == 4: # both
        both += 1


model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='multi:softmax', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,class_num=4,
       silent=True, subsample=1)

#--------------------------------
#####    MULTICLASS MODEL   #####
#--------------------------------
print("MULTICLASS MODEL")
print("Step 1 : train the model")
model.fit(features_train, classification_train)
print("Model trained")
# print(model)
# print()
# print(model.feature_importances_)

# make predictions for test data
print("Step 2 : predict the model")
y_pred = model.predict(features_test)
print("Model Predicted")
print("Step 3 : Extract predictions")
predictions = [round(value) for value in y_pred]
print("Predictions Extracted")

#creation de la matrice de confusion
mc = 5*[0]
for i in range(len(mc)): mc[i] =5*[0]
t=0
#comme en python les tableaux exsitent pas, je decide de la parcourir ligne après ligne
#dans l'ordre : n, c, w, b
#attention les liste commencent à 0 et non pas à 1!


cc = tc = cw = tw = cb = tb = cn = tn = 0
for i in range(0,len(classification_test)):
    vr = int(classification_test[i])
    vp = int(predictions[i])

    mc[vr-1][vp-1]+=1
    mc[vr-1][4]+=1
    mc[4][vp-1]+=1
    t+=1

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
mc[4][4] = t
verif=(mc[4][0]+mc[4][1]+mc[4][2]+mc[4][3])-(mc[0][4]+mc[1][4]+mc[2][4]+mc[3][4])
print("voici la matrice : ")
print(mc)
print("elle est juste si %d=0" % (verif))

# evaluate predictions
accuracy = sklm.accuracy_score(classification_test, predictions)
accuracy_nb = sklm.accuracy_score(classification_test, predictions,normalize=False)

SEc = cc/tc
SEw = cw/tw
SEb = cb/tb
print("Sensitivity crackles (cc/tc) : %.4f" % SEc)
print("Sensitivity wheezes (cw/tw) : %.4f" % SEw)
print("Sensitivity both (cb/tb) : %.4f" % SEb)
sensitivity_num = cc+cw+cb
sensitivity_denum = tc+tw+tb
sensitivity_global = sensitivity_num/sensitivity_denum
specificity_global = cn/tn
accuracy = (sensitivity_global + specificity_global)/2
print("sensitivity global : %.4f" % sensitivity_global)
print("specificity (cn/tn)) : %.4f" % specificity_global)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')
