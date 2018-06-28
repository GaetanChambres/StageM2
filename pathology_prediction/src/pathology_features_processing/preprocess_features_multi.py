import csv
import sys

def file_len(fname):
    with open(fname) as f:
        nb = 0
        while(f.readline()):
            nb+=1
    return nb

input1 = './pathology_prediction/data/csv/challenge/train_all.csv'
input2 = './pathology_prediction/data/csv/challenge/train_info.csv'
output = './pathology_prediction/data/csv/challenge/train_pathologies_multi.csv'

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
    feat = features.readline()
    infos = info.readline()
    if(cpt==0):
        out.write(file+","+"pathologie"+","+rest)
        cpt = 1
    else:
        out.write(name+","+patho[-2]+","+feats[:-1])
