import numpy as np
import pandas as pd

train_DS1 = pd.read_csv('./data/csv/mfccs_csv/DS1features.csv', header = None)
test_DS1 = pd.read_csv('./data/csv/mfccs_csv/DS1features.csv', skiprows = 2, header = None) # Make sure to skip a row for the test set

tmp = "filename,startTime,endTime,"
for i in range(1,14):
    tmp += "coef"+str(i)+","
for j in range(1,41):
    tmp += "bands"+str(j)+","
tmp += "crackle,wheezle"
cols_labels = tmp.split(",")

train_DS1.columns = cols_labels
test_DS1.columns = cols_labels
test_DS1.info()
