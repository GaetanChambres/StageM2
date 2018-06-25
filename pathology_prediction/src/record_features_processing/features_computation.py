import essentia.standard as es
import os
import numpy as np

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

input_directory = './pathology_prediction/data/database/challenge/train/'
output_directory = './pathology_prediction/data/csv/record_level/'

output_file = 'train_all'
output = output_directory + output_file
out_file = open(output+".csv", "w")
display_ui = "********************************************"

cpt_file = 0
nb_files = (len(os.listdir(input_directory)))/2
ordered_files = sorted(os.listdir(input_directory))

array_features_size = []
index_array = 0

for filename in ordered_files:

    if(filename.endswith('.txt')):
        cpt_file += 1
        input = input_directory + filename
        input_file = open(input,"r")
        file = filename[:-4]
        audio_in = input_directory + file + ".wav"
        print("processing file %d / %d" % (cpt_file,nb_files))

        stats = ['min', 'max', 'median', 'mean', 'var', 'stdev', 'dmean', 'dvar', 'dmean2', 'dvar2']
        # mfccStats = ['mean', 'cov', 'icov']
        # gfccStats = ['mean', 'cov', 'icov']

        profile = {
            # 'startTime': float(start_time),
            # 'endTime': float(end_time),
            'lowlevelFrameSize': 2048,
            'lowlevelHopSize': 1024,
            'lowlevelWindowType': 'blackmanharris62',
            'lowlevelSilentFrames': 'noise',
            'lowlevelStats': stats,

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

        tab_header = []
        tab_value = []

        features_name = sorted(features.descriptorNames())
        for feat in range(0,len(features_name)):
            name_feature = features_name[feat]
            # print(name)
            kind_of_feature = name_feature.split(".")
            # print(type)

            # ALL FEATURES
            if(kind_of_feature[0] != 'metadata' and
            kind_of_feature[1] != 'silence_rate_60dB' and
            kind_of_feature[1] != 'sound_start_frame' and
            kind_of_feature[1] != 'sound_stop_frame' and
            kind_of_feature[1] != 'beats_position' and
            kind_of_feature[1] != 'bpm_intervals' and
            kind_of_feature[1] != 'onset_times' and
            kind_of_feature[1] != 'chords_key' and
            kind_of_feature[1] != 'chords_progression' and
            kind_of_feature[1] != 'chords_scale' and
            kind_of_feature[1] != 'key_edma' and
            kind_of_feature[1] != 'key_krumhansl' and
            kind_of_feature[1] != 'key_temperley'):

            # LOWLEVEL FEATURES
            # if(kind_of_feature[0] != 'metadata' and
            # kind_of_feature[0] == 'lowlevel' and
            # kind_of_feature[1] != 'silence_rate_60dB' and
            # kind_of_feature[1] != 'sound_start_frame' and
            # kind_of_feature[1] != 'sound_stop_frame'):

            # RHYTHM FEATURES
            # if(kind_of_feature[0] != 'metadata' and
            # kind_of_feature[0] == 'rhythm' and
            # kind_of_feature[1] != 'beats_position' and
            # kind_of_feature[1] != 'bpm_intervals' and
            # kind_of_feature[1] != 'onset_times' and):

            # SFX FEATURES
            # if(kind_of_feature[0] != 'metadata' and
            # kind_of_feature[0] == 'sfx'):

            # TONAL FEATURES
            # if(kind_of_feature[0] != 'metadata' and
            # kind_of_feature[0] == 'tonal' and
            # kind_of_feature[1] != 'chords_key' and
            # kind_of_feature[1] != 'chords_progression' and
            # kind_of_feature[1] != 'chords_scale' and
            # kind_of_feature[1] != 'key_edma' and
            # kind_of_feature[1] != 'key_krumhansl' and
            # kind_of_feature[1] != 'key_temperley'):

            # MFCC
            # if(kind_of_feature[1] == "mfcc" and kind_of_feature[2] == "mean"):


                tmp = features[name_feature]
                # print(type(tmp))
                if(type(tmp) is np.ndarray):
                    dim = tmp.shape
                    if(len(dim) == 2):
                        # dimension
                        cpt=0
                        for i in range(0,dim[1]) :
                            for j in range(0,dim[0]) :
                                tab_header.append(name_feature+"_"+str(i)+"_"+str(j))
                                tab_value.append(tmp[i,j])
                    else:
                        dim = tmp.shape
                        for k in range(0,dim[0]):
                            tab_header.append(name_feature+"_"+str(k))
                            tab_value.append(tmp[k])
                else:
                    tab_header.append(name_feature)
                    tab_value.append(tmp)


        # print(len(tab_header))
        # print(len(tab_value))
        # print(tab_header)
        # print(tab_value)

        header = "filename,"
        vals = str(file)+","
        for h in range(0,len(tab_header)):
            header += str(tab_header[h])+","
        for v in range(0,len(tab_value)):
            vals += str(tab_value[v])+","

        header = header[:-1]
        vals = vals[:-1]
        print(len(header.split(",")))
        print(len(vals.split(",")))

        if cpt_file == 1 :
            out_file.write(str(header))
        out_file.write("\n")
        out_file.write(str(vals))

        content = input_file.readline()
