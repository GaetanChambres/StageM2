#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
# from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

def get_confusion_matrix (c1, c2, n_clusters) :
    confusion_mat = np.zeros((n_clusters, n_clusters))

    for class_num in range(n_clusters) :
        for i in range(len(c1)):
            classed = c1[i]

    for i in range(n_clusters) :
        c_i = set([index for index, value in enumerate(c1) if value == i])
        for j in range(n_clusters) :
            c_j = set([index for index, value in enumerate(c2) if value == j])

            confusion_mat[i, j] =len(c_i.intersection(c_j))

    return confusion_mat

input_train = "./pathology_prediction/data/csv/record_level/train_pathologies.csv"
input_test = "./pathology_prediction/data/csv/record_level/test_pathologies.csv"

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
features_train = train_dataset[:,1:]

pathologies_test = test_dataset[:,0:1]
features_test = test_dataset[:,1:]
print("Numpy arrays ready")
######################################################################################################
print("Preparing data for DMatrix")
print("Preparing train data")
train_dataset = pd.read_csv(input_train)
pathologies_T = train_dataset.pathologie
filenames_T = train_dataset.filename
# print(pathologies_T)
del train_dataset['filename']
features_T = train_dataset.drop(['pathologie'],axis=1)
# print(features_T)
print("Train data ready")
print("Preparing test data")
test_dataset = pd.read_csv(input_test)
pathologies_t = test_dataset.pathologie
filenames_t = test_dataset.filename
# print(pathologies_t)
del test_dataset['filename']
features_t = test_dataset.drop(['pathologie'],axis=1)
# print(features_t)
print("Test data ready")
print("Converting data to DMatrix")
nb_lines = len(test_dataset)
asthma = LRTI = pneumonia = bronchioectasis = bronchiolitis = URTI = COPD = healthy = total = 0
ratios = 1
train = xgb.DMatrix(features_T,label=pathologies_T)
test = xgb.DMatrix(features_t,label=pathologies_t)
print("DMatrix ready")
print("#################")
print("Computing the ratios for balanced training")
print("Parsing train data")
for i in range(0,len(pathologies_train)):
    if(pathologies_train[i] == 0):
        asthma += 1
        total += 1
    if(pathologies_train[i] == 1):
        LRTI += 1
        total += 1
    if(pathologies_train[i] == 2):
        pneumonia += 1
        total += 1
    if(pathologies_train[i] == 3):
        bronchioectasis += 1
        total += 1
    if(pathologies_train[i] == 4):
        bronchiolitis += 1
        total += 1
    if(pathologies_train[i] == 5):
        URTI += 1
        total += 1
    if(pathologies_train[i] == 6):
        COPD += 1
        total += 1
    if(pathologies_train[i] == 7):
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
print("Compute the ratios")
if(asthma == 0) : ratio_asthma = 0
else : ratio_asthma = (total-asthma) / asthma
ratio_LRTI = (total-LRTI) / LRTI
ratio_pneumonia = (total-pneumonia) / pneumonia
ratio_bronchioectasis = (total-bronchioectasis) / bronchioectasis
ratio_bronchiolitis = (total-bronchiolitis) / bronchioectasis
ratio_URTI = (total-URTI) / URTI
ratio_COPD = (total-COPD) / COPD
ratio_healthy = (total-healthy) / healthy

ratios = [ratio_asthma,ratio_LRTI,ratio_pneumonia,ratio_bronchioectasis,ratio_bronchiolitis,ratio_URTI,ratio_COPD,ratio_healthy]
print("Ratios are computed :")
print(ratios)
print("#################")


# model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#        learning_rate=0.1, max_delta_step=0,
#     , n_estimators=100,
#        reg_alpha=0, reg_lambda=1, scale_pos_weight=ratios, seed=None,class_num=8,
#        silent=True, subsample=1)
params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
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
xgb.plot_importance(model,max_num_features = 25)
from matplotlib import pyplot
pyplot.show()
print()
###################################################################################
# make predictions for test data
print("Step 2 : predict the model")
y_pred = model.predict(test)
labels = test.get_label()
print("Model Predicted")
print("Step 3 : Extract predictions")
print(y_pred.shape)
print(y_pred)

output_directory = './pathology_prediction/data/results/'
output_file = 'results_records_proba.csv'
output = output_directory + output_file
out_file = open(output, "w")
out_file.write("file,asthma,LRTI,pneumonia,bronchioectasis,bronchiolitis,URTI,COPD,healthy\n")
for i in range(len(y_pred)):
    out_file.write(filenames_t[i]+",")
    for j in range(0,8):
        out_file.write(str(y_pred[i][j])+",")
    # print("writing csv file")
    out_file.write(str(pathologies_t[i]+1))
    out_file.write("\n")
print('file write ok')
