import os
import essentia
import matplotlib.pyplot as plt
import essentia.standard as std
from essentia.standard import *

# init folders to work on
input_data = './data/database/version3_challenge/test/'
info_file = './data/csv/challenge/test_info.csv'
output_file = './data/csv/challenge/mfcc_test.csv'

with open(info_file) as f:
    for i, l in enumerate(f):
        pass
        tmp = i + 1
    nb_lines = tmp

in_info = open(info_file, "r")
out_file = open(output_file, "w")
cpt = 0
# read global csv
# 1 line is 1 respiration cycle in 1 record
# we want to compute features for each ones
info = in_info.readline() #line 1
while info:
    cpt+=1
    #saving csv info of the line in some variables
    patient_number,record_index,body_area,record_tool,channel,patho,start_time,end_time,crackle,wheezle = info.split(',')

    # starting the  features computation

    tmpfile = input_data+patient_number+"_"+record_index+"_"+body_area+"_"+channel+"_"+record_tool+".wav"
    #print tmpfile
    loader = essentia.standard.EasyLoader(filename=tmpfile, startTime=float(start_time), endTime=float(end_time))
    audio = loader()

    # init essentia algorithms
    w = Windowing(type = 'hann')
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = MFCC()

    # compute mfcc using frame generator
    mfccs = []
    melbands = []
    melbands_log = []

    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        logNorm = UnaryOperator(type='log')
        melbands_log.append(logNorm(mfcc_bands))

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = essentia.array(mfccs).T
    melbands = essentia.array(melbands).T
    melbands_log = essentia.array(melbands_log).T



    string_mfcc = ""
    for t in range(len(mfcc_coeffs)):
        string_mfcc += str(mfcc_coeffs[t]) + ","


    # for t in range(len(mfcc_bands)):
    #     string_mfcc += str(mfcc_bands[t]) + ","
    #
    # string_mfcc = string_mfcc[:-1]
    # #print string_mfcc
    #
    classification = 0
    # if normal -> classification 1 / if crackles -> classification 2 / if wheezles -> classification 3 / if both -> classification 4
    if(int(crackle) == 0 and int(wheezle) == 0):
        classification = 1
    elif(int(crackle) == 1 and int(wheezle) == 1):
        classification = 4
    else:
        if(int(crackle) == 1):
            classification = 2
        if(int(wheezle) == 1):
            classification = 3
    # write mfccs file
    # out_file.write(patient_number + "," + start_time + "," + end_time + "," + str(normal) + "," + crackle + "," + wheezle.rstrip('\n') + "," + str(both) + "," + string_mfcc[:-1] + "\n")
    out_file.write(patient_number + "," + start_time + "," + end_time + "," + str(classification) + "," + string_mfcc[:-1] + "\n")

    # out_file.write("\n")
    print str(cpt) + " over " + str(nb_lines)
    info = in_info.readline()
