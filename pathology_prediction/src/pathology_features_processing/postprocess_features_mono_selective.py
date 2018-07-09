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
output = './pathology_prediction/data/csv/challenge/mono_class_selective/vs_healthy/train_asthma_mono.csv'

# input1 = './pathology_prediction/data/csv/challenge/test_all.csv'
# input2 = './pathology_prediction/data/csv/challenge/test_info.csv'
# output = './pathology_prediction/data/csv/challenge/mono_class_selective/vs_healthy/test_asthma_mono.csv'

 #######################
 # pathology goes from 0 to 6
pathology_to_work_on = 0
cpt_true = 0
cpt_false = 0
cpt_total = 0
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
print(feat)
splitted = feat.split(',',2)
filename = splitted[0]
classif = splitted[1]
print(filename)
print(classif)

infos = info.readline()
print(infos)
cpt=0
while(feat and infos):
    splitted = feat.split(',',2)
    filename = splitted[0]
    classif = splitted[1]

    values = feat.split(',')
    name = values[0]
    feats = ""
    for i in range(1,len(values)):
        feats += (str(values[i])+",")

    filename,patho = infos.split(",")
    patho = int(patho)

    if(cpt==0):
        out.write(file+","+"pathologie"+","+rest)
        cpt = 1
#######################
    # if pathology_to_work_on == 7 and patho == pathology_to_work_on: #working on normal cycles
    #     cpt_true += 1
    #     patho = 1
    if patho == 7:
        patho = 0
        if cpt == 1:
            out.write(name+","+str(patho)+","+feats[:-1])
            cpt_false +=1
    elif patho == pathology_to_work_on and int(classif) !=1:
        cpt_true += 1
        patho = 1
        if cpt == 1:
            out.write(name+","+str(patho)+","+feats[:-1])
    else:
        patho = 0
        # cpt_false += 1
        # if cpt == 1:
        #     out.write(name+","+str(patho)+","+feats[:-1])
    cpt_total+=1
    print("%d true occurences / %d false occurences / %d total occurences"%(cpt_true,cpt_false,cpt_total))
#######################
    feat = features.readline()
    infos = info.readline()
