#!/usr/bin/python3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score

##############################
#####     Open Files     #####
##############################

#####   train   #####
train_mfcc = pd.read_csv('./data/csv/mfccs_csv/DS1features.csv', header=1)
train_fileinfo = pd.read_csv('./data/csv/initial_csv/DS1global.csv', header=1)

#####   test    #####
test_mfcc = pd.read_csv('./data/csv/mfccs_csv/DS3features.csv', header=1)
test_fileinfo = pd.read_csv('./data/csv/initial_csv/DS3global.csv', header=1)


##############################
#  Naming cols in csv files  #
##############################
tmp = "filename,startTime,endTime,"

#####   info files  #####
tmp_info = tmp + "crackles,wheezles"
cols_labels_info = tmp_info.split(",")
#print(cols_labels_info)
train_fileinfo.columns = cols_labels_info
test_fileinfo.columns = cols_labels_info
# DEBUG: prints
#train_fileinfo.info()
#print(train_fileinfo.head())
#print(train_fileinfo.shape)
#
#test_fileinfo.info()
#print(test_fileinfo.head())
#print(test_fileinfo.shape)

##### mfcc files   #####
tmp_mfcc = tmp
for i in range(1,14):
    tmp_mfcc += "coef"+str(i)+","
for j in range(1,41):
    tmp_mfcc += "bands"+str(j)+","
tmp_mfcc = tmp_mfcc[:-1]
cols_labels_mfcc = tmp_mfcc.split(",")
#print cols_labels_mfcc
train_mfcc.columns = cols_labels_mfcc
test_mfcc.columns = cols_labels_mfcc
# DEBUG: prints
#train_mfcc.info()
#print(train_mfcc.head())
#print(train_mfcc.shape)

#test_mfcc.info()
#print(test_mfcc.head())
#print(test_mfcc.shape)


##############################
# Filtering data for dataset #
##############################

#####   train   #####
dataset_train_mfcc = train_mfcc[[col for col in train_mfcc.columns if col!='filename' and col!='startTime' and col!= 'endTime']]
# print(dataset_train_mfcc.shape)

crackles_train = train_fileinfo[[col for col in train_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='wheezles']]
wheezles_train = train_fileinfo[[col for col in train_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='crackles']]
# print(crackles_train.shape)
# print(wheezles_train.shape)

#####   test    #####
dataset_test_mfcc = test_mfcc[[col for col in test_mfcc.columns if col!='filename' and col!='startTime' and col!= 'endTime']]
# print(dataset_test_mfcc.shape)

crackles_test = test_fileinfo[[col for col in test_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='wheezles']]
wheezles_test = test_fileinfo[[col for col in test_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='crackles']]
# print(crackles_test.shape)
# print(wheezles_test.shape)

##################################################
#####          Training and predictions      #####
##################################################

#####   Model params    #####
param = {
    'max_depth': 50,  # the maximum depth of each tree
    'eta': 1,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 2,  # the number of classes that exist in this datset
    'eval_metric': 'mlogloss' } #the eval metric used

num_round = 20  # the number of training iterations

#--------------------------------
#####   Working on crackles #####
#--------------------------------
crackles_matrix_train = xgb.DMatrix(dataset_train_mfcc, label=crackles_train)
crackles_matrix_test = xgb.DMatrix(dataset_test_mfcc, label=crackles_test)

# training and testing - numpy matrices
crackles_trained_model = xgb.train(param, crackles_matrix_train, num_round)
crackles_predictions = crackles_trained_model.predict(crackles_matrix_test)

# extracting most confident predictions
crackles_best_preds = np.asarray([np.argmax(line) for line in crackles_predictions])
print("Precision of prediction for crackles: " + str(precision_score(crackles_test, crackles_best_preds, average='macro')) + " over 1.0")

#--------------------------------
#####   Working on wheezles #####
#--------------------------------
wheezles_matrix_train = xgb.DMatrix(dataset_train_mfcc, label=wheezles_train)
wheezles_matrix_test = xgb.DMatrix(dataset_test_mfcc, label=wheezles_test)

# training and testing - numpy matrices
wheezles_trained_model = xgb.train(param, wheezles_matrix_train, num_round)
wheezles_predictions = wheezles_trained_model.predict(wheezles_matrix_test)

# extracting most confident predictions
wheezles_best_preds = np.asarray([np.argmax(line) for line in wheezles_predictions])
print("Precision of prediction for wheezles: " + str(precision_score(wheezles_test, wheezles_best_preds, average='macro')) + " over 1.0")
