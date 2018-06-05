import essentia
import essentia.standard as es
import yaml

input = open("data/database/version2_patient/DS1/102_1b1_Ar_sc_AKGC417L.txt")
start,end,crackles,wheezles = input.readline().split()

audio_in = "data/database/version2_patient/DS1/102_1b1_Ar_sc_AKGC417L.wav"
features_out = "data/tmp/features_test.json"


profile = dict(
    startTime = start,
    endTime =  end,
    outputFrames = "0",
    outputFormat = "json",
    requireMbid = "false",
    indent = "4",

    lowlevel = dict(
        frameSize = "2048",
        hopSize = "1024",
        zeroPadding = "0",
        windowType = "blackmanharris62",
        silentFrames = "noise",
        stats = "[\"mean\", \"var\", \"median\", \"min\", \"max\", \"dmean\", \"dmean2\", \"dvar\", \"dvar2\"]"),

    rhythm = dict(
        method = "degara",
        minTempo = "40",
        maxTempo = "208",
        stats = "[\"mean\", \"var\", \"median\", \"min\", \"max\", \"dmean\", \"dmean2\", \"dvar\", \"dvar2\"]"),

    tonal = dict(
        frameSize = "4096",
        hopSize= "2048",
        zeroPadding = "0",
        windowType = "blackmanharris62",
        silentFrames = "noise",
        stats = "[\"mean\", \"var\", \"median\", \"min\", \"max\", \"dmean\", \"dmean2\", \"dvar\", \"dvar2\"]"))

with open('profile.yml', 'w') as outfile:
    yaml.dump(profile, outfile, default_flow_style=False)

import sys
import subprocess
arg1 = audio_in
arg2 = features_out
subprocess.call(['essentia_streaming_extractor_freesound', arg1, arg2, 'profile.yml'])

print(start)

import json
with open("features_test.json") as json_data:
    d = json.load(json_data)
    print(d)
