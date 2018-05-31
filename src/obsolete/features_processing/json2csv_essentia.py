#!/usr/bin/env python3
import json
import csv
from flatten_json import flatten

with open('features_test.json') as data_file:
    json = json.load(data_file)

    data = flatten(json)
    # data = json.dumps(data_tmp, sort_keys=True)

values = ""
features = ""
with open('features_test.csv', 'w') as csv_file:
    for key, value in sorted(data.items()):
        # writer.writerow([key, value])
        features += str(key) + ","
        values += str(value) + ","
    csv_file.write(str(features))
    csv_file.write("\n")
    csv_file.write(str(values))
