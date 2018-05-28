#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
import csv as csv
# from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

final_array_patient_problem = []
final_array_patient_ok = []

input = open("./data/csv/challenge/final.csv","r")
output = open("./data/csv/challenge/final_res.csv","w")
line = input.readline()
patient = ""
nb_cycle = nb_cycle_patient = normal = crackle = wheeze = both = ratio_c = ratio_w = ratio_b = ratio_n = 0
while(line):
    file,classif = line.split(",")
    if(file[:3] != patient):
        if(patient != ""):
            print("For patient number " + patient)
            print("nb_cycle = %d" % nb_cycle_patient)
            print("normal = %d" % normal)
            print("crackle = %d" % crackle)
            print("wheeze = %d" % wheeze)
            print("both = %d" % both)

            if crackle != 0:
                print('This patient has crackles')
                ratio_c = crackle / nb_cycle
                print('Ratio of crackles is %f' % ratio_c)
            if wheeze != 0:
                print('This patient has wheezes')
                ratio_w = wheeze / nb_cycle
                print('Ratio of wheezes is %f' % ratio_w)
            if both != 0:
                print('This patient has both crackles and wheezes at the same time')
                ratio_b = both / nb_cycle
                print('Ratio of both is %f' % ratio_b)
            if normal != 0:
                ratio_n = normal / nb_cycle
                print('Ratio of normals is %f' % ratio_n)
            if(crackle == 0 and wheeze == 0 and both == 0):
                print('This patient has no particularity in his respiration')
            print("(debug :: total of %% = %.2f)" % (ratio_n+ratio_c+ratio_w+ratio_b))
            print()
            if(ratio_c > 0.1 or ratio_w > 0.1 or ratio_b > 0.1):
                final_array_patient_problem.append(patient)
                pb = 1
            else:
                final_array_patient_ok.append(patient)
                pb = 0
            if ratio_n > ratio_c+ratio_w+ratio_b:
                majority = 1
            elif ratio_c > ratio_n+ratio_w+ratio_b:
                majority = 2
            elif ratio_w > ratio_n+ratio_c+ratio_b:
                majority = 3
            elif ratio_b > ratio_n+ratio_c+ratio_w:
                majority = 4
            res = patient+","+str(round(ratio_n, 2))+','+str(round(ratio_c, 2))+','+str(round(ratio_w, 2))+','+str(round(ratio_b, 2))+','+str(majority)+','+str(pb)
            output.write(res)
            output.write("\n")

        nb_cycle = 0
        normal = crackle = wheeze = both = nb_cycle_patient = 0
        ratio_n = ratio_c = ratio_w = ratio_b = 0
        patient = file[:3]
    elif file[:3] == patient :
        nb_cycle+=1
        nb_cycle_patient = nb_cycle
        if(int(classif) == 1):
            normal+=1
        elif(int(classif) == 2):
            crackle+=1
        elif(int(classif) == 3):
            wheeze+=1
        elif(int(classif) == 4):
            both+=1
    else:
        print("ERROR while reading patient number")
    line = input.readline()

print('patient with problems : %d' % len(final_array_patient_problem))
print(final_array_patient_problem)
print()
print('patient with no problems %d' % len(final_array_patient_ok))
print(final_array_patient_ok)
