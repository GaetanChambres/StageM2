import os
import csv
import re


input_directory = './data/database/version3_challenge/test/'
output_directory = './data/csv/challenge/'
output_file = 'test_info.csv'
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

        content = input_file.readline()
        tmp = filename[:-4]
        patient_number,record_index,body_area,channel,record_tool = tmp.split("_")

        # patient_number = filename[:3]
        # record_index = filename[4:7]
        # body_area = filename[8:10]
        # channel = filename[11:13]
        # record_tool = filename[14:-4]

        # print(filename)
        # print(patient_number)
        # print(record_index)
        # print(body_area)
        # print(channel)
        # print(record_tool)

        while content:
            start_time,end_time,crackle,wheezle = content.split('\t')
            # print "start_time = ",start_time
            # print "end_time = ",end_time
            # print "crackles = ",crackle
            # print "wheezle = ",wheezle

            # print(patient_number + "," + record_index + "," + body_area + "," + record_tool + "," + channel + "," + start_time + "," + end_time + "," + crackle + "," + wheezle)
            out_file.write(patient_number + "," + record_index + "," + body_area + "," + record_tool + "," + channel + "," + start_time + "," + end_time + "," + crackle + "," + wheezle)

            content = input_file.readline()
        print "processed ",cpt,"over ",nb_files
