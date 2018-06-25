#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

def file_len(input):
    with open(input) as f1:
        line1 = f1.readline()
        data1= line1.split(',')
        # print(data1)
        nbcols = len(data1)
        # print(nbcols)
    return nbcols

DS1 = "./pathology_prediction/data/csv/patient/DS1/DS1_pathologies.csv"
DS2 = "./pathology_prediction/data/csv/patient/DS2/DS2_pathologies.csv"
DS3 = "./pathology_prediction/data/csv/patient/DS3/DS3_pathologies.csv"

print("Computing len files")
DS1len = file_len(DS1)
DS2len = file_len(DS2)
DS3len = file_len(DS3)

print("Loading data files")
DS1_dataset = np.loadtxt(DS1, delimiter=",", skiprows=1, usecols=range(1,DS1len))
DS2_dataset = np.loadtxt(DS2, delimiter=",", skiprows=1, usecols=range(1,DS2len))
DS3_dataset = np.loadtxt(DS3, delimiter=",", skiprows=1, usecols=range(1,DS3len))
print("data files loaded")
print("append train files")
# ------------------------------------------------------
# train_dataset = np.append(DS1_dataset,DS2_dataset,axis=0)
# test_dataset = DS3_dataset
# ------------------------------------------------------
# train_dataset = np.append(DS2_dataset,DS3_dataset,axis=0)
# test_dataset = DS1_dataset
# ------------------------------------------------------
train_dataset = np.append(DS1_dataset,DS3_dataset,axis=0)
test_dataset = DS2_dataset
# ------------------------------------------------------
print("split data in arrays")
pathologies_train = train_dataset[:,0:1]
classification_train = train_dataset[:,1:2]
features_train = train_dataset[:,2:]

pathologies_test = test_dataset[:,0:1]
classification_test = test_dataset[:,1:2]
features_test = test_dataset[:,2:]
print("data splitted in arrays")

print("parsing data for ratio computation")
asthma = LRTI = pneumonia = bronchioectasis = bronchiolitis = URTI = COPD = healthy = total = 0
for i in range(0,len(pathologies_train)):
    if(pathologies_train[i] == 1):
        asthma += 1
        total += 1
    if(pathologies_train[i] == 2):
        LRTI += 1
        total += 1
    if(pathologies_train[i] == 3):
        pneumonia += 1
        total += 1
    if(pathologies_train[i] == 4):
        bronchioectasis += 1
        total += 1
    if(pathologies_train[i] == 5):
        bronchiolitis += 1
        total += 1
    if(pathologies_train[i] == 6):
        URTI += 1
        total += 1
    if(pathologies_train[i] == 7):
        COPD += 1
        total += 1
    if(pathologies_train[i] == 8):
        healthy += 1
        total += 1
print(asthma)
print(LRTI)
print(pneumonia)
print(bronchioectasis)
print(bronchiolitis)
print(URTI)
print(COPD)
print(healthy)
print(total)
if(asthma == 0):
    ratio_asthma = 0
else:
    ratio_asthma = (total-asthma) / asthma
ratio_LRTI = (total-LRTI) / LRTI
ratio_pneumonia = (total-pneumonia) / pneumonia
ratio_bronchioectasis = (total-bronchioectasis) / bronchioectasis
ratio_bronchiolitis = (total-bronchiolitis) / bronchioectasis
ratio_URTI = (total-URTI) / URTI
ratio_COPD = (total-COPD) / COPD
ratio_healthy = (total-healthy) / healthy
print("data parsed for ratio computation")
ratios = [ratio_asthma,ratio_LRTI,ratio_pneumonia,ratio_bronchioectasis,ratio_bronchiolitis,ratio_URTI,ratio_COPD,ratio_healthy]
print(ratios)

model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='multi:softmax', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=ratios, seed=None,#class_num=8,
       silent=True, subsample=1)

print("MULTICLASS MODEL")
print("Step 1 : train the model")
model.fit(features_train, pathologies_train)
print("Model trained")
print("Step 2 : predict the model")
y_pred = model.predict(features_test)
print("Model Predicted")
print("Step 3 : Extract predictions")
predictions = [round(value) for value in y_pred]
print("Predictions Extracted")
print(y_pred)

p1 = c1 = t1 = p2 = c2 = t2 = p3 = c3 = t3 = p4 = c4 = t4 = 0
p5 = c5 = t5 = p6 = c6 = t6 = p7 = c7 = t7 = p8 = c8 = t8 = 0
arr = 9*[0]
total = 0
total_ok = 0
for i in range(len(arr)) : arr[i] = 3* [0]

for i in range(0,len(pathologies_test)):
    vr = int(pathologies_test[i])
    vp = int(predictions[i])

    if vr == 1:
        t1+=1
        arr[vr-1][0]+=1
        if vp == 1:
            total_ok +=1
            c1+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 2:
        t2+=1
        arr[vr-1][0]+=1
        if vp == 2:
            total_ok +=1
            c2+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 3:
        t3+=1
        arr[vr-1][0]+=1
        if vp == 3:
            total_ok +=1
            c3+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 4:
        t4+=1
        arr[vr-1][0]+=1
        if vp == 4:
            total_ok +=1
            c4+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 5:
        t5+=1
        arr[vr-1][0]+=1
        if vp == 5:
            total_ok +=1
            c5+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 6:
        t6+=1
        arr[vr-1][0]+=1
        if vp == 6:
            total_ok +=1
            c6+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 7:
        t7+=1
        arr[vr-1][0]+=1
        if vp == 7:
            total_ok +=1
            c7+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    if vr == 8:
        t8+=1
        arr[vr-1][0]+=1
        if vp == 8:
            total_ok +=1
            c8+=1
            arr[vp-1][1]+=1
            arr[vp-1][2]+=1
        else:
            arr[vp-1][1]+=1
    total+=1

arr[8][0] = len(pathologies_test)
arr[8][1] = total
arr[8][2] = total_ok

print( "a triplet by pathology \n [ a, b, c ] \n a = nb of cycle expected \n  b = nb total of cycle predicted \n b = nb of cycle correctly prediced \n")
print(arr)
print()
print("c1 = %d, t1 = %d, c2 = %d, t2 = %d, c3 = %d, t3 = %d, c4 = %d, t4 = %d" % (c1,t1,c2,t2,c3,t3,c4,t4))
print("c5 = %d, t5 = %d, c6 = %d, t6 = %d, c7 = %d, t7 = %d, c8 = %d, t8 = %d" % (c5,t5,c6,t6,c7,t7,c8,t8))
