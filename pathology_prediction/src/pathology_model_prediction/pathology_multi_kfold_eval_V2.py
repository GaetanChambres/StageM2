#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

input_total = "./pathology_prediction/data/csv/complete/complete_pathologies.csv"

with open(input_total) as f3:
    line3 = f3.readline()
    data3 = line3.split(',')
    print(data3)
    nbcols_total = len(data3)
    print(nbcols_total)

print("Opening data as Numpy array")
print("Loading data")
total_dataset = np.loadtxt(input_total, delimiter=",", skiprows=1, usecols=range(1,nbcols_total))
print("Parsing data")
pathologies = total_dataset[:,0:1]
classification = total_dataset[:,1:2]
features = total_dataset[:,2:]

nb_lines = len(total_dataset)

seed = 1
test_size_ratio = 0.33
features_train, features_test, pathologies_train, pathologies_test = train_test_split(features, pathologies, test_size = test_size_ratio, random_state = seed)
print("Data ready")

print("Computation of the ratios")
asthma = LRTI = pneumonia = bronchioectasis = bronchiolitis = URTI = COPD = healthy = total = 0
print("init ok !")
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

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 9,
    'gamma': 0,
    'max_depth': 3,
    'colsample_bytree': 1,
    'min_child_weight': 1,
    'silent': 1,
    'eta': 0.1,
    'scale_pos_weight': ratios,
}

num_round = 50

print("Preparing kfold")
train=pd.read_csv(input_total)
feature_names = list(train)
del feature_names[0]
del feature_names[1]
del feature_names[2]
print(feature_names)
from sklearn.model_selection import KFold
del train['filename']
del train['classification']
k = 3
kfold = KFold(n_splits=k)
features_T = train.drop(['pathologie'],axis=1).values
pathologies_T = train.pathologie
print("Data ready for Kfold")
print("Kfold test starting")
for i, (train_index, test_index) in enumerate(kfold.split(features_T,pathologies_T)):
    print('[Fold %d/%d]' % (i+1,3))
    X_train, X_test = features_T[train_index], features_T[test_index]
    y_train, y_test = pathologies_T[train_index], pathologies_T[test_index]
    d_train = xgb.DMatrix(X_train,label=y_train,feature_names=feature_names)
    d_test = xgb.DMatrix(X_test,label=y_test,feature_names=feature_names)
    watchlist = [(d_test, 'eval'), (d_train, 'train')]
    print("Training")
    evals_result = {}
    model = xgb.train(params, d_train, num_round, watchlist, evals_result=evals_result)
    print("Trained")
    xgb.plot_importance(model,max_num_features = 25)
    from matplotlib import pyplot
    pyplot.show()
    print()

    print("Predictions")
    y_pred = model.predict(d_test)
    labels = d_test.get_label()
    print("Model Predicted")
    print("Step 3 : Extract predictions")
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
    confusion = sklm.confusion_matrix(y_test,y_pred)
    print(len(confusion))
    print(confusion)
print("Kfold test finished")
