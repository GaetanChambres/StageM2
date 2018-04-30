#!/usr/bin/env python3
import json
import csv
from flatten_json import flatten

with open('features_test.json') as data_file:
    data = json.load(data_file)

    data = flatten(data)

    # print(data)

with open('features_test.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in data.items():
       writer.writerow([key, value])


with open('features_test.csv', 'r') as f:
    for i, l in enumerate(f):
        pass
        tmp = i + 1
    nb_lines = tmp
# header = ""
# values = []
in_csv = open("features_test.csv", "r")
info = in_csv.readline() #line 1
feature = []
value = []


# while info:
#     f,v = info.split(",")
#     feature.append(f)
#     value.append(v)
#     info = in_csv.readline()
#
# print(len(feature))
# print(len(value))
# print(feature)
# print(value)
