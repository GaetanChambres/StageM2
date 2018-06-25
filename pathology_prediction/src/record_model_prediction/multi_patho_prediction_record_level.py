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
# print(pathologies_T)
del train_dataset['filename']
features_T = train_dataset.drop(['pathologie'],axis=1)
# print(features_T)
print("Train data ready")
print("Preparing test data")
test_dataset = pd.read_csv(input_test)
pathologies_t = test_dataset.pathologie
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
    'objective': 'multi:softmax',
    'num_class': 8,
    'gamma': 0,
    'max_depth': 3,
    'subsample': 0.7,
    'colsample_bytree': 1,
    'min_child_weight': 1,
    'silent': 1,
    'eta': 0.1,
    'seed': 1,
}
num_round = 10
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
# print(model)
# xgb.plot_importance(model)
# from matplotlib import pyplot
# pyplot.show()
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
confusion = sklm.confusion_matrix(pathologies_test,y_pred)
best_preds = np.asarray([np.argmax(line) for line in y_pred])
print(confusion)
p1 = c1 = t1 = p2 = c2 = t2 = p3 = c3 = t3 = p4 = c4 = t4 = 0
p5 = c5 = t5 = p6 = c6 = t6 = p7 = c7 = t7 = p8 = c8 = t8 = 0
arr = 9*[0]
total = 0
total_ok = 0
for i in range(len(arr)) : arr[i] = 3* [0]

for i in range(0,len(pathologies_test)):
    vr = int(pathologies_test[i])
    vp = int(y_pred[i])

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
