#!/usr/bin/env python3
import os
import csv
import re
import yaml
import json
import csv
from flatten_json import flatten

input_directory = './data/database/DS1Full/'
output_directory = './data/csv/initial_csv_v3/'
output_file = 'DS1globalV3.csv'
output = output_directory + output_file
out_file = open(output, "w")

cpt = 0
nb_files = (len(os.listdir(input_directory)))/2
ordered_files = sorted(os.listdir(input_directory))

for filename in ordered_files:
    if(filename.endswith('.txt')):
        cpt = cpt+1
        input = input_directory + filename
        input_file = open(input,"r")

        file = filename[:-4]
        audio_in = input_directory + file + ".wav"

        patient_number,record_index,body_area,channel,record_tool = file.split("_")

        content = input_file.readline()
        while content:
            start_time,end_time,crackle,wheezle = content.split('\t')
            
            profile = dict(
                startTime = start_time,
                endTime =  end_time,
                outputFrames = "0",
                outputFormat = "json",
                requireMbid = "false",
                indent = "4",

                lowlevel = dict(
                    frameSize = "2048",
                    hopSize = "1024",
                    zeroPadding = "0",
                    windowType = "blackmanharris62",
                    silentFrames = "noise",
                    stats = "[\"mean\", \"var\", \"median\", \"min\", \"max\", \"dmean\", \"dmean2\", \"dvar\", \"dvar2\"]"),

                rhythm = dict(
                    method = "degara",
                    minTempo = "40",
                    maxTempo = "208",
                    stats = "[\"mean\", \"var\", \"median\", \"min\", \"max\", \"dmean\", \"dmean2\", \"dvar\", \"dvar2\"]"),

                tonal = dict(
                    frameSize = "4096",
                    hopSize= "2048",
                    zeroPadding = "0",
                    windowType = "blackmanharris62",
                    silentFrames = "noise",
                    stats = "[\"mean\", \"var\", \"median\", \"min\", \"max\", \"dmean\", \"dmean2\", \"dvar\", \"dvar2\"]"))

            with open('profile.yml', 'w') as outfile:
                yaml.dump(profile, outfile, default_flow_style=False)

            import sys
            import subprocess
            arg1 = audio_in
            arg2 = "tmp.json"
            subprocess.call(['essentia_streaming_extractor_freesound', arg1, arg2])

            import json
            import csv
            from flatten_json import flatten

            with open(arg2) as data_file:
                json = json.load(data_file)

                data = flatten(json)
            values = ""
            features = ""

            for key, value in sorted(data.items()):
                # writer.writerow([key, value])
                features += str(key) + ","
                values += str(value) + ","
            if cpt == 1 :
                out_file.write(str(features))
            out_file.write("\n")
            out_file.write(file+","+str(values))

            content = input_file.readline()
        print("processed ",cpt,"over ",nb_files)
