#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

# input_train = "./pathology_prediction/data/csv/debug/train_pathologies.csv"
# input_test = "./pathology_prediction/data/csv/debug/test_pathologies.csv"

input_train = "./pathology_prediction/data/csv/challenge/train_pathologies.csv"
input_test = "./pathology_prediction/data/csv/challenge/test_pathologies.csv"
input_total = "./pathology_prediction/data/csv/complete/total_pathologies.csv"

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
# header = train_dataset[:1,:]
pathologies_train = train_dataset[:,0:1]
classification_train = train_dataset[:,1:2]
features_train = train_dataset[:,2:]

pathologies_test = test_dataset[:,0:1]
classification_test = test_dataset[:,1:2]
features_test = test_dataset[:,2:]

nb_lines = len(test_dataset)
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
print("computation ratios")
ratio_asthma = (total-asthma) / asthma
ratio_LRTI = (total-LRTI) / LRTI
ratio_pneumonia = (total-pneumonia) / pneumonia
ratio_bronchioectasis = (total-bronchioectasis) / bronchioectasis
ratio_bronchiolitis = (total-bronchiolitis) / bronchioectasis
ratio_URTI = (total-URTI) / URTI
ratio_COPD = (total-COPD) / COPD
ratio_healthy = (total-healthy) / healthy

ratios = [ratio_asthma,ratio_LRTI,ratio_pneumonia,ratio_bronchioectasis,ratio_bronchiolitis,ratio_URTI,ratio_COPD,ratio_healthy]
print(ratios)

print("computation is over")


model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='multi:softmax', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=ratios, seed=None,class_num=8,
       silent=True, subsample=1)

#--------------------------------
#####    MULTICLASS MODEL   #####
#--------------------------------
# sample_weights_data = [0,0,0.04,0.01,0.02,0.03,0.83,0.05]
# sample_weights_data = [0,0,0,0,0,0,1,0]
print("MULTICLASS MODEL")
print("Step 1 : train the model")
# model.fit(features_train, classification_train)
model.fit(features_train, pathologies_train)
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
print(y_pred)
# #creation du tableau cycles/pathology
# pat = 5*[0] #5 lines
# for i in range(len(pat)): pat[i] = 9*[0] #9 cols
# t2=0
# #creation de la matrice de confusion
# mc = 5*[0] #5 lines
# for i in range(len(mc)): mc[i] = 5*[0] #5 cols
# t=0


p1 = c1 = t1 = p2 = c2 = t2 = p3 = c3 = t3 = p4 = c4 = t4 = 0
p5 = c5 = t5 = p6 = c6 = t6 = p7 = c7 = t7 = p8 = c8 = t8 = 0
arr = 9*[0]
total = 0
total_ok = 0
for i in range(len(arr)) : arr[i] = 3* [0]
# for i in range(0,len(pathologies_train)):
#     vr = int(pathologies_train[i])
#     nb_pat = int(pathologies_train[i])
#
#     pat[vr-1][nb_pat-1]+=1
#     pat[vr-1][8]+=1
#     pat[4][nb_pat-1]+=1
#     t2+=1

for i in range(0,len(pathologies_test)):
    vr = int(pathologies_test[i])
    vp = int(predictions[i])
    # nb_pat = int(pathologies_test[i])
    #
    # pat[vr-1][nb_pat-1]+=1
    # pat[vr-1][8]+=1
    # pat[4][nb_pat-1]+=1
    # t2+=1
    #
    # mc[vr-1][vp-1]+=1
    # mc[vr-1][4]+=1
    # mc[4][vp-1]+=1
    # t+=1

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
# mc[4][4] = t
# pat[4][8] = t2
# verif=(mc[4][0]+mc[4][1]+mc[4][2]+mc[4][3])-(mc[0][4]+mc[1][4]+mc[2][4]+mc[3][4])
# print("tabeleau :")
# print(pat)
# print("voici la matrice : ")
# print(mc)
# print("elle est juste si %d=0" % (verif))

# evaluate predictions
accuracy = sklm.accuracy_score(pathologies_test, predictions)
accuracy_nb = sklm.accuracy_score(pathologies_test, predictions,normalize=False)
#
# SE1 = c1/t1
# SE2 = c2/t2
# SE3 = c3/t3
# SE4 = c4/t4
# SE5 = c5/t5
# SE6 = c6/t6
# SE7 = c7/t7
# SE8 = c8/t8
# print("Sensitivity asthma (c1/t1) : %.4f" % SE1)
# print("Sensitivity LRTI (c2/t2) : %.4f" % SE2)
# print("Sensitivity Pneumonia (c3/t3) : %.4f" % SE3)
# print("Sensitivity Bronchioectasis (c4/t4) : %.4f" % SE4)
# print("Sensitivity Bronchiolitis (c5/t5) : %.4f" % SE5)
# print("Sensitivity URTI (c6/t6) : %.4f" % SE6)
# print("Sensitivity COPD (c7/t7) : %.4f" % SE7)
# print("Sensitivity Healthy (c8/t8) : %.4f" % SE8)
# sensitivity_num = c1+c2+c3+c4+c5+c6+c7
# sensitivity_denum = t1+t2+t3+t4+t5+t6+t7
# sensitivity_global = sensitivity_num/sensitivity_denum
# specificity_global = c8/t8
# accuracy = (sensitivity_global + specificity_global)/2
# print("sensitivity global : %.4f" % sensitivity_global)
# print("specificity (cn/tn)) : %.4f" % specificity_global)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print('*********************')
