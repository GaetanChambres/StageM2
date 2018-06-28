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

input_train = "./pathology_prediction/data/csv/challenge/mono_class/train_healthy_mono.csv"
input_test = "./pathology_prediction/data/csv/challenge/mono_class/test_healthy_mono.csv"

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

######################################################################################################
#
print("Opening data as Numpy array")
print("Loading data")
train_dataset = np.loadtxt(input_train, delimiter=",", skiprows = 1, usecols=range(1,nbcols_train))
test_dataset = np.loadtxt(input_test, delimiter=",", skiprows = 1, usecols=range(1,nbcols_test))
print("Parsing data")
pathologies_train = train_dataset[:,0:1]
classification_train = train_dataset[:,1:2]
features_train = train_dataset[:,2:]

pathologies_test = test_dataset[:,0:1]
classification_test = test_dataset[:,1:2]
features_test = test_dataset[:,2:]
print("Numpy arrays ready")
######################################################################################################
print("Preparing data for DMatrix")
print("Preparing train data")
train_dataset = pd.read_csv(input_train)
pathologies_T = train_dataset.pathologie
# print(pathologies_T)
del train_dataset['filename']
del train_dataset['classification']
features_T = train_dataset.drop(['pathologie'],axis=1)
# print(features_T)
print("Train data ready")
print("Preparing test data")
test_dataset = pd.read_csv(input_test)
pathologies_t = test_dataset.pathologie
# print(pathologies_t)
del test_dataset['filename']
del test_dataset['classification']
features_t = test_dataset.drop(['pathologie'],axis=1)
# print(features_t)
print("Test data ready")
print("Converting data to DMatrix")
nb_lines = len(test_dataset)
negatives = 0
positives = 0
ratios = 1
train = xgb.DMatrix(features_T,label=pathologies_T)
test = xgb.DMatrix(features_t,label=pathologies_t)
print("DMatrix ready")
print("#################")
print("Computing the ratios for balanced training")
print("Parsing train data")
for i in range(0,len(pathologies_train)):
    if(pathologies_train[i] == 0):
        negatives += 1
    if(pathologies_train[i] == 1):
        positives += 1
print("Compute the ratios")
if(positives == 0):
    ratios = 0
else :
    ratios = negatives / positives
print("Ratios are computed :")
print(ratios)
print("#################")

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 8,
    'gamma': 0,
    'max_depth': 3,
    'colsample_bytree': 1,
    'min_child_weight': 1,
    'silent': 1,
    'eta': 0.1,
    'scale_pos_weight': ratios,
}
num_round = 50
watchlist = [(test, 'eval'), (train, 'train')]
#--------------------------------
#####    MULTICLASS MODEL   #####
#--------------------------------
# sample_weights_data = [0,0,0.04,0.01,0.02,0.03,0.83,0.05]
# sample_weights_data = [0,0,0,0,0,0,1,0]
print("MULTICLASS MODEL")
print("Step 1 : train the model")
# model.fit(features_train, classification_train)
# model.fit(features_train, pathologies_train)
evals_result = {}
model = xgb.train(params, train, num_round, watchlist, evals_result=evals_result)
print("Model trained")
print()
# print(model.feature_importances_)
# print(model)
xgb.plot_importance(model,max_num_features = 25)
from matplotlib import pyplot
# xgb.plot_tree(model)
pyplot.show()
print()
###################################################################################
# make predictions for test data
print("Step 2 : predict the model")
y_pred = model.predict(test)
labels = test.get_label()
print("Model Predicted")
print("Step 3 : Extract predictions")
# predictions = [round(value) for value in y_pred]
print("Predictions Extracted")
print('error=%f' % (sum(1 for i in range(len(y_pred)) if int(y_pred[i] > 0.5) != labels[i]) / float(len(y_pred))))
print(evals_result)
print("***********")
test_eval= pathologies_test.flatten()
print(len(y_pred))
print(len(test_eval))
print(type(y_pred))
print(y_pred)
print(type(test_eval))
print(test_eval)
print("***********")
# confusion = get_confusion_matrix(test_eval,y_pred,8)
confusion1 = sklm.confusion_matrix(pathologies_test,y_pred)
print(len(confusion1))
print(confusion1)
# best_preds = np.asarray([np.argmax(line) for line in y_pred])
# confusion2 = sklm.confusion_matrix(pathologies_test,best_preds)
