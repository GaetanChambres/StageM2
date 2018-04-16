import os
import csv


input_directory = './data/database/DS3/'
output_directory = './data/csv/initial_csv'
output_file = 'DS3global.csv'
output = output_directory + output_file
out_file = open(output, "w")

cpt = 0
nb_files = (len(os.listdir(input_directory)))/2

for filename in os.listdir(input_directory):
    if(filename.endswith('.txt')):
        cpt = cpt+1
        input = input_directory + filename

        input_file = open(input,"r")

        content = input_file.readline()

        while content:
            start_time,end_time,crackle,wheezle = content.split('\t')
            #print "start_time = ",start_time
            #print "end_time = ",end_time
            #print "crackles = ",crackle
            #print "wheezle = ",wheezle

            out_file.write(filename[:-4] + "," + start_time + "," + end_time + "," + crackle + "," + wheezle)
            content = input_file.readline()
        print "processed ",cpt,"over ",nb_files
