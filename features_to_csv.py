import os
import essentia
import matplotlib.pyplot as plt
import essentia.standard as std
from essentia.standard import *

# init folders to work on
input_data = './data/database/DS1/'
info_file = './data/csv/DS1global.csv'
output_file = './data/csv/mfccs_csv/DS1features.csv'

in_info = open(info_file, "r")
out_file = open(output_file, "w")

# read global csv
# 1 line is 1 respiration cycle in 1 record
# we want to compute features for each ones
info = in_info.readline() #line 1
while info:
    #saving csv info of the line in some variables
    filename,start_time,end_time,crackle,wheezle = info.split(',')
    start_time = float(start_time)
    end_time = float(end_time)
    crackle = int(crackle)
    wheezle = int(wheezle)

    # starting the  features computation

    tmpfile = input_data+filename+".wav"
    #print tmpfile
    loader = essentia.standard.EasyLoader(filename=tmpfile, startTime=start_time, endTime=end_time)
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

    # and plot
    # plt.imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
    # plt.title("Mel band spectral energies in frames")
    # plt.show()
    #
    # plt.imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
    # plt.title("Log-normalized mel band spectral energies in frames")
    # plt.show()
    #
    # plt.imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
    # plt.title("MFCCs in frames")
    # plt.show()

    #print mfccs


    break
    info = in_info.readline()
