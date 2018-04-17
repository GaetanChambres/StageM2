import numpy as np
import pandas as pd

train_DS1_mfcc = pd.read_csv('./data/csv/mfccs_csv/DS1features.csv', header = None)
train_DS1_info = pd.read_csv('./data/csv/initial_csv/DS1global.csv', header = None)

test_DS1_mfcc = pd.read_csv('./data/csv/mfccs_csv/DS1features.csv', header = None)
test_DS1_info = pd.read_csv('./data/csv/initial_csv/DS1global.csv', header = None)

tmp = "filename,startTime,endTime,"
tmp_mfcc = tmp
tmp_info = tmp + "crackles,wheezles"
for i in range(1,14):
    tmp_mfcc += "coef"+str(i)+","
for j in range(1,41):
    tmp_mfcc += "bands"+str(j)+","
tmp_mfcc = tmp_mfcc[:-1]
cols_labels_mfcc = tmp_mfcc.split(",")
cols_labels_info = tmp_info.split(",")

print cols_labels_mfcc
print cols_labels_info

train_DS1_mfcc.columns = cols_labels_mfcc
test_DS1_mfcc.columns = cols_labels_mfcc

train_DS1_info.columns = cols_labels_info
test_DS1_info.columns = cols_labels_info

train_DS1_mfcc.info()
test_DS1_mfcc.info()
train_DS1_info.info()
test_DS1_info.info()
