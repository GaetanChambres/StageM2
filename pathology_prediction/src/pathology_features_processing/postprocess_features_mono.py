import csv
import sys

def file_len(fname):
    with open(fname) as f:
        nb = 0
        while(f.readline()):
            nb+=1
    return nb

input1 = './pathology_prediction/data/csv/challenge/test_all.csv'
input2 = './pathology_prediction/data/csv/challenge/test_info.csv'
output = './pathology_prediction/data/csv/challenge/test_pathologies_mono.csv'

 #######################
 # pathology goes from 1 to 8 : asthma | pneumonia
pathology_to_work_on = 6
cpt_true = 0
cpt_false = 0
 #######################

feat_len = file_len(input1)-1
info_len = file_len(input2)

if((feat_len)!=info_len):
    print("ERROR ! Invalid files")
    sys.exit()

features = open(input1,'r')
info = open(input2,'r')
out = open(output,'w')


header = features.readline()
file = header[:8]
rest = header[9:]

feat = features.readline()
infos = info.readline()
cpt=0
while(feat and infos):

    values = feat.split(',')
    name = values[0]
    feats = ""
    for i in range(1,len(values)):
        feats += (str(values[i])+",")

    filename,patho = infos.split(",")
#######################

    if int(patho) == pathology_to_work_on :
        patho = 1
        cpt_true += 1
    else :
        patho = 0
        cpt_false += 1
    print("%d true occurences / %d false occurences / %d total occurences"%(cpt_true,cpt_false,cpt_true+cpt_false))
#######################
    feat = features.readline()
    infos = info.readline()
    if(cpt==0):
        out.write(file+","+"pathologie"+","+rest)
        cpt = 1
    else:
        out.write(name+","+str(patho)+","+feats[:-1])
