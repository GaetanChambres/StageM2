#!/usr/bin/env python3
from __future__ import print_function

import numpy as np
import librosa
import sys
import argparse

def mfcc_calc(input_file, output_csv):

    fd = open(output_csv, "w")

    filename = input_file
    filename = filename[-10:-4]
    i = 5
    while filename[0]=='0':
        filename = filename[-i:]
        i = i - 1

    fd.write('%s,' % filename)

    features = []

    y, sr = librosa.load(input_file, sr = 44100)
    hop_length = 1024

    mfcc = librosa.feature.mfcc(y = y, sr = sr)

    for coefficient in mfcc:
        features.append(np.mean(coefficient))
        fd.write('%f,' % np.mean(coefficient))

    fd.close()

if __name__ == '__main__':

    mfcc_calc(sys.argv[1], sys.argv[2])
