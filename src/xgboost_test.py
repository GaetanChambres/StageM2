#!/usr/bin/python3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score

##############################
#####     Open Files     #####
##############################

DS1_features = pd.read_csv('./data/csv/mfccs_csv/DS1features.csv', header=None)
DS1_fileinfo = pd.read_csv('./data/csv/initial_csv/DS1global.csv', header=None)

DS2_features = pd.read_csv('./data/csv/mfccs_csv/DS2features.csv', header=None)
DS2_fileinfo = pd.read_csv('./data/csv/initial_csv/DS2global.csv', header=None)

DS3_features = pd.read_csv('./data/csv/mfccs_csv/DS3features.csv', header=None)
DS3_fileinfo = pd.read_csv('./data/csv/initial_csv/DS3global.csv', header=None)

#####   train   #####
train_mfcc = DS1_features
train_fileinfo = DS1_fileinfo

train_mfcc_2 = DS2_features
train_fileinfo_2 = DS2_fileinfo

#####   test    #####
test_mfcc = DS3_features
test_fileinfo = DS3_fileinfo

##############################
#  Naming cols in csv files  #
##############################
tmp = "filename,startTime,endTime,"

#####   info files  #####
tmp_info = tmp + "crackles,wheezles"
cols_labels_info = tmp_info.split(",")
#print(cols_labels_info)
train_fileinfo.columns = cols_labels_info
train_fileinfo_2.columns = cols_labels_info
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
train_mfcc_2.columns = cols_labels_mfcc
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
print("TRAIN inputs")
dataset_train_mfcc = train_mfcc[[col for col in train_mfcc.columns if col!='filename' and col!='startTime' and col!= 'endTime']]
dataset_train_mfcc_2 = train_mfcc_2[[col for col in train_mfcc_2.columns if col!='filename' and col!='startTime' and col!= 'endTime']]
combined_train_mfcc = pd.concat([dataset_train_mfcc, dataset_train_mfcc_2],axis=0,ignore_index=True) # Stacks them vertically
print("shape of data input 1:",dataset_train_mfcc.shape)
print("shape of data input 2:",dataset_train_mfcc_2.shape)
print("shape of combined data input:",combined_train_mfcc.shape)
print()

crackles_train = train_fileinfo[[col for col in train_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='wheezles']]
crackles_train_2 = train_fileinfo_2[[col for col in train_fileinfo_2.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='wheezles']]
combined_crackles_train = pd.concat([crackles_train, crackles_train_2],axis=0,ignore_index=True)
print("shape of crackles results 1:",crackles_train.shape)
print("shape of crackles results 2:",crackles_train_2.shape)
print("shape of combined crackles results:",combined_crackles_train.shape)
print()

wheezles_train = train_fileinfo[[col for col in train_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='crackles']]
wheezles_train_2 = train_fileinfo_2[[col for col in train_fileinfo_2.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='crackles']]
combined_wheezles_train = pd.concat([wheezles_train, wheezles_train_2],axis=0,ignore_index=True)
print("shape of wheezles results 1:",wheezles_train.shape)
print("shape of wheezles results 2:",wheezles_train_2.shape)
print("shape of combined wheezles results:",combined_wheezles_train.shape)
print()

#####   test    #####
print("TEST inputs")
dataset_test_mfcc = test_mfcc[[col for col in test_mfcc.columns if col!='filename' and col!='startTime' and col!= 'endTime']]
print("shape of data input",dataset_test_mfcc.shape)

crackles_test = test_fileinfo[[col for col in test_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='wheezles']]
wheezles_test = test_fileinfo[[col for col in test_fileinfo.columns if col!='filename' and col!='startTime' and col!= 'endTime' and col!='crackles']]
print("shape of crackles results",crackles_test.shape)
print("shape of wheezles results",wheezles_test.shape)
print()

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
crackles_matrix_train = xgb.DMatrix(combined_train_mfcc, label=combined_crackles_train)
#crackles_matrix_train = xgb.DMatrix(dataset_train_mfcc, label=crackles_train)
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
wheezles_matrix_train = xgb.DMatrix(combined_train_mfcc, label=combined_wheezles_train)
#wheezles_matrix_train = xgb.DMatrix(dataset_train_mfcc, label=wheezles_train)
wheezles_matrix_test = xgb.DMatrix(dataset_test_mfcc, label=wheezles_test)

# training and testing - numpy matrices
wheezles_trained_model = xgb.train(param, wheezles_matrix_train, num_round)
wheezles_predictions = wheezles_trained_model.predict(wheezles_matrix_test)

# extracting most confident predictions
wheezles_best_preds = np.asarray([np.argmax(line) for line in wheezles_predictions])
print("Precision of prediction for wheezles: " + str(precision_score(wheezles_test, wheezles_best_preds, average='macro')) + " over 1.0")
