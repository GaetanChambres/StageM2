import csv
import sys

def file_len(fname):
    with open(fname) as f:
        nb = 0
        while(f.readline()):
            nb+=1
    return nb


# input1 = './pathology_prediction/data/csv/challenge/train_all.csv'
# input2 = './pathology_prediction/data/csv/challenge/train_info.csv'
# output = './pathology_prediction/data/csv/challenge/mono_class/train_healthy_mono.csv'

input1 = './pathology_prediction/data/csv/challenge/train_all.csv'
input2 = './cycle_prediction/data/csv/challenge/train_info.csv'
# output = './pathology_prediction/data/csv/challenge/mono_class/test_healthy_mono.csv'

 #######################
 # pathology goes from 0 to 7
pathology_to_work_on = 4
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
# out = open(output,'w')


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

    patient_number,patho,record_index,body_area,channel,record_tool,start,stop,cr,wh = infos.split(",")
#######################
    wh = wh[0]
    classif = 0
    # print("****")
    # print(cr)
    # print(wh)
    # print("****")
    if(cr == "0" and wh == "0"):
        classif = 1
    if(cr == "1" and wh == "0"):
        classif = 2
    if(cr == "0" and wh == "1"):
        classif = 3
    if(cr == "1" and wh == "1"):
        classif = 4

    if int(classif) == pathology_to_work_on :
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
        # out.write(file+","+"pathologie"+","+rest)
        cpt = 1
    else:
        # out.write(name+","+str(classif)+","+feats[:-1])
        cpt+=1
