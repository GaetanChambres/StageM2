import essentia.standard as es
import os

# **********************************
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
# **********************************

input_directory = './data/database/version3_challenge/test/'
output_directory = './data/csv/challenge/'
output_file = 'features_test'
output = output_directory + output_file
out_file = open(output+".csv", "w")
display_ui = "********************************************"

cpt_file = 0
nb_files = (len(os.listdir(input_directory)))/2
ordered_files = sorted(os.listdir(input_directory))

for filename in ordered_files:

    if(filename.endswith('.txt')):
        cpt_file += 1
        input = input_directory + filename
        input_file = open(input,"r")
        nb_cycles = file_len(input)
        file = filename[:-4]
        audio_in = input_directory + file + ".wav"

        patient_number,record_index,body_area,channel,record_tool = file.split("_")
        cpt_cycle = 0
        content = input_file.readline()
        while content:
            cpt_cycle += 1
            print(display_ui)
            print("processing cycle %d / %d in the file %d / %d" % (cpt_cycle,nb_cycles,cpt_file,nb_files))
            print(display_ui)
            start_time,end_time,crackle,wheezle = content.split('\t')

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

            stats = ['min', 'max', 'median', 'mean', 'var', 'stdev', 'dmean', 'dvar', 'dmean2', 'dvar2']
            # mfccStats = ['mean', 'cov', 'icov']
            # gfccStats = ['mean', 'cov', 'icov']

            profile = {
                'startTime': float(start_time),
                'endTime': float(end_time),
                'lowlevelFrameSize': 2048,
                'lowlevelHopSize': 1024,
                'lowlevelWindowType': 'blackmanharris62',
                'lowlevelSilentFrames': 'noise',
                'lowlevelStats': stats,

                # 'loudnessFrameSize': 88200,
                # 'loudnessHopSize': 44100,
                # 'loudnessWindowType': 'hann',
                # 'loudnessSilentFrames': 'noise',

                'rhythmMethod': 'degara',
                'rhythmMinTempo': 40,
                'rhythmMaxTempo': 208,
                'rhythmStats': stats,

                'tonalFrameSize': 4096,
                'tonalHopSize': 2048,
                'tonalWindowType': 'blackmanharris62',
                'tonalSilentFrames': 'noise',
                'tonalStats': stats
            }

            features, features_frames = es.FreesoundExtractor(**profile)(audio_in)
            file_json = output+".json"

            es.YamlOutput(filename=file_json, format='json')(features)

            import json
            import csv
            import subprocess
            from pandas.io.json import json_normalize
            # from flatten_json import flatten
            arg1 = 's/inf/Infinity/g'
            arg2 = file_json
            cmd = str("sed -i " + arg1 + " " + arg2)
            os.system(cmd)
            with open(file_json) as data_file:
                json = json.load(data_file)
                data = flatten_json(json)
                json_normalize(data)
            values = ""
            features = ""

            for key, value in sorted(data.items()):
                # writer.writerow([key, value])
                features += str(key) + ","
                values += str(value) + ","
            if cpt_file == 1:
                header = "filename" + ",classif," + str(features)
                out_file.write(str(header))
            out_file.write("\n")
            out_file.write(file + "," + str(classification) + "," + str(values))

            content = input_file.readline()
        print(display_ui)
        print("file processed %d / %d" % (cpt_file,nb_files))
        print(display_ui)
out_file.close()
