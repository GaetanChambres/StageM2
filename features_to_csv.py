import os
import essentia
import matplotlib.pyplot as plt
import essentia.standard as std
from essentia.standard import *

# init folders to work on
input_data = './data/database/DS3/'
info_file = './data/csv/initial_csv/DS3global.csv'
output_file = './data/csv/mfccs_csv/DS3features.csv'

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
    filename,start_time,end_time,crackle,wheezle = info.split(',')

    # starting the  features computation

    tmpfile = input_data+filename+".wav"
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


    # write mfccs file
    out_file.write(filename + "," + start_time + "," + end_time + ",")

    string_mfcc = ""
    for t in range(len(mfcc_coeffs)):
        string_mfcc += str(mfcc_coeffs[t]) + ","

    #print string_mfcc

    for t in range(len(mfcc_bands)):
        string_mfcc += str(mfcc_bands[t]) + ","

    string_mfcc += crackle + "," + wheezle

    print string_mfcc
    out_file.write(string_mfcc)

    print str(cpt) + " over " + str(nb_lines)
    info = in_info.readline()
