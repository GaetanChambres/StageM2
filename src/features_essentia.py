import essentia.standard as es
import os
import numpy as np

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

input_directory = './data/database/debug/train/'
output_directory = './data/csv/debug/'
output_file = 'train_debug'
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
        nb_cycles = file_len(input)
        file = filename[:-4]
        audio_in = input_directory + file + ".wav"

        # patient_number,record_index,body_area,channel,record_tool = file.split("_")
        cpt_cycle = 0
        content = input_file.readline()
        while content:
            cpt_cycle += 1
            index_array += 1
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

            tab_header = []
            tab_value = []
            
            features_name = sorted(features.descriptorNames())
            for feat in range(0,len(features_name)):
                name_feature = features_name[feat]
                # print(name)
                kind_of_feature = name_feature.split(".")
                # print(type)
                if(kind_of_feature[0] == 'lowlevel'):
                    tmp = features[name_feature]
                    # print(type(tmp))
                    if(type(tmp) is np.ndarray):
                        # print("simple dim ??")
                        # print(name_feature)
                        dim = tmp.shape
                        # print(len(dim))
                        if(len(dim) == 2):
                            # print("double dim ??")
                            print(name_feature)
                            len2 = len(tmp[0])
                            len1 = len(tmp)
                            print(len1)
                            print(len2)
                            cpt=0
                            for i in range(0,len1) :

                                for j in range(0,len2) :
                                    tab_header.append(name_feature+"_"+str(i)+"_"+str(j))
                                    tab_value.append(tmp[i,j])
                        else:
                            dim = tmp.shape
                            len1 = dim[0]
                            for i in range(0,len1):
                                tab_header.append(name_feature+"_"+str(i))
                                tab_value.append(tmp[i])
                    else:
                        tab_header.append(name_feature)
                        tab_value.append(tmp)


            print(len(tab_header))
            print(len(tab_value))
            print(tab_header)




            # print("info de :lowlevel.average_loudness", features['lowlevel.average_loudness'])
            # print()
            # print("info de :lowlevel.barkbands.dmean", len(features['lowlevel.barkbands.dmean']))
            # print("info de :lowlevel.barkbands.dmean2", len(features['lowlevel.barkbands.dmean2']))
            # print("info de :lowlevel.barkbands.dvar", len(features['lowlevel.barkbands.dvar']))
            # print("info de :lowlevel.barkbands.dvar2", len(features['lowlevel.barkbands.dvar2']))
            # print("info de :lowlevel.barkbands.max", len(features['lowlevel.barkbands.max']))
            # print("info de :lowlevel.barkbands.mean", len(features['lowlevel.barkbands.mean']))
            # print("info de :lowlevel.barkbands.median", len(features['lowlevel.barkbands.median']))
            # print("info de :lowlevel.barkbands.min", len(features['lowlevel.barkbands.min']))
            # print("info de :lowlevel.barkbands.stdev", len(features['lowlevel.barkbands.stdev']))
            # print("info de :lowlevel.barkbands.var", len(features['lowlevel.barkbands.var']))
            # print()
            # print("info de :lowlevel.barkbands_crest.dmean", features['lowlevel.barkbands_crest.dmean'])
            # print("info de :lowlevel.barkbands_crest.dmean2", features['lowlevel.barkbands_crest.dmean2'])
            # print("info de :lowlevel.barkbands_crest.dvar", features['lowlevel.barkbands_crest.dvar'])
            # print("info de :lowlevel.barkbands_crest.dvar2", features['lowlevel.barkbands_crest.dvar2'])
            # print("info de :lowlevel.barkbands_crest.max", features['lowlevel.barkbands_crest.max'])
            # print("info de :lowlevel.barkbands_crest.mean", features['lowlevel.barkbands_crest.mean'])
            # print("info de :lowlevel.barkbands_crest.median", features['lowlevel.barkbands_crest.median'])
            # print("info de :lowlevel.barkbands_crest.min", features['lowlevel.barkbands_crest.min'])
            # print("info de :lowlevel.barkbands_crest.stdev", features['lowlevel.barkbands_crest.stdev'])
            # print("info de :lowlevel.barkbands_crest.var", features['lowlevel.barkbands_crest.var'])
            # print()
            # print("info de :lowlevel.barkbands_flatness_db.dmean", features['lowlevel.barkbands_flatness_db.dmean'])
            # print("info de :lowlevel.barkbands_flatness_db.dmean2", features['lowlevel.barkbands_flatness_db.dmean2'])
            # print("info de :lowlevel.barkbands_flatness_db.dvar", features['lowlevel.barkbands_flatness_db.dvar'])
            # print("info de :lowlevel.barkbands_flatness_db.dvar2", features['lowlevel.barkbands_flatness_db.dvar2'])
            # print("info de :lowlevel.barkbands_flatness_db.max", features['lowlevel.barkbands_flatness_db.max'])
            # print("info de :lowlevel.barkbands_flatness_db.mean", features['lowlevel.barkbands_flatness_db.mean'])
            # print("info de :lowlevel.barkbands_flatness_db.median", features['lowlevel.barkbands_flatness_db.median'])
            # print("info de :lowlevel.barkbands_flatness_db.min", features['lowlevel.barkbands_flatness_db.min'])
            # print("info de :lowlevel.barkbands_flatness_db.stdev", features['lowlevel.barkbands_flatness_db.stdev'])
            # print("info de :lowlevel.barkbands_flatness_db.var", features['lowlevel.barkbands_flatness_db.var'])
            # print()
            # print("info de :lowlevel.barkbands_kurtosis.dmean", features['lowlevel.barkbands_kurtosis.dmean'])
            # print("info de :lowlevel.barkbands_kurtosis.dmean2", features['lowlevel.barkbands_kurtosis.dmean2'])
            # print("info de :lowlevel.barkbands_kurtosis.dvar", features['lowlevel.barkbands_kurtosis.dvar'])
            # print("info de :lowlevel.barkbands_kurtosis.dvar2", features['lowlevel.barkbands_kurtosis.dvar2'])
            # print("info de :lowlevel.barkbands_kurtosis.max", features['lowlevel.barkbands_kurtosis.max'])
            # print("info de :lowlevel.barkbands_kurtosis.mean", features['lowlevel.barkbands_kurtosis.mean'])
            # print("info de :lowlevel.barkbands_kurtosis.median", features['lowlevel.barkbands_kurtosis.median'])
            # print("info de :lowlevel.barkbands_kurtosis.min", features['lowlevel.barkbands_kurtosis.min'])
            # print("info de :lowlevel.barkbands_kurtosis.stdev", features['lowlevel.barkbands_kurtosis.stdev'])
            # print("info de :lowlevel.barkbands_kurtosis.var", features['lowlevel.barkbands_kurtosis.var'])
            # print()
            # print("info de :lowlevel.barkbands_skewness.dmean", features['lowlevel.barkbands_skewness.dmean'])
            # print("info de :lowlevel.barkbands_skewness.dmean2", features['lowlevel.barkbands_skewness.dmean2'])
            # print("info de :lowlevel.barkbands_skewness.dvar", features['lowlevel.barkbands_skewness.dvar'])
            # print("info de :lowlevel.barkbands_skewness.dvar2", features['lowlevel.barkbands_skewness.dvar2'])
            # print("info de :lowlevel.barkbands_skewness.max", features['lowlevel.barkbands_skewness.max'])
            # print("info de :lowlevel.barkbands_skewness.mean", features['lowlevel.barkbands_skewness.mean'])
            # print("info de :lowlevel.barkbands_skewness.median", features['lowlevel.barkbands_skewness.median'])
            # print("info de :lowlevel.barkbands_skewness.min", features['lowlevel.barkbands_skewness.min'])
            # print("info de :lowlevel.barkbands_skewness.stdev", features['lowlevel.barkbands_skewness.stdev'])
            # print("info de :lowlevel.barkbands_skewness.var", features['lowlevel.barkbands_skewness.var'])
            # print()
            # print("info de :lowlevel.barkbands_spread.dmean", features['lowlevel.barkbands_spread.dmean'])
            # print("info de :lowlevel.barkbands_spread.dmean2", features['lowlevel.barkbands_spread.dmean2'])
            # print("info de :lowlevel.barkbands_spread.dvar", features['lowlevel.barkbands_spread.dvar'])
            # print("info de :lowlevel.barkbands_spread.dvar2", features['lowlevel.barkbands_spread.dvar2'])
            # print("info de :lowlevel.barkbands_spread.max", features['lowlevel.barkbands_spread.max'])
            # print("info de :lowlevel.barkbands_spread.mean", features['lowlevel.barkbands_spread.mean'])
            # print("info de :lowlevel.barkbands_spread.median", features['lowlevel.barkbands_spread.median'])
            # print("info de :lowlevel.barkbands_spread.min", features['lowlevel.barkbands_spread.min'])
            # print("info de :lowlevel.barkbands_spread.stdev", features['lowlevel.barkbands_spread.stdev'])
            # print("info de :lowlevel.barkbands_spread.var", features['lowlevel.barkbands_spread.var'])
            # print()
            # print("info de :lowlevel.dissonance.dmean", features['lowlevel.dissonance.dmean'])
            # print("info de :lowlevel.dissonance.dmean2", features['lowlevel.dissonance.dmean2'])
            # print("info de :lowlevel.dissonance.dvar", features['lowlevel.dissonance.dvar'])
            # print("info de :lowlevel.dissonance.dvar2", features['lowlevel.dissonance.dvar2'])
            # print("info de :lowlevel.dissonance.max", features['lowlevel.dissonance.max'])
            # print("info de :lowlevel.dissonance.mean", features['lowlevel.dissonance.mean'])
            # print("info de :lowlevel.dissonance.median", features['lowlevel.dissonance.median'])
            # print("info de :lowlevel.dissonance.min", features['lowlevel.dissonance.min'])
            # print("info de :lowlevel.dissonance.stdev", features['lowlevel.dissonance.stdev'])
            # print("info de :lowlevel.dissonance.var", features['lowlevel.dissonance.var'])
            # print()
            # print("info de :lowlevel.dynamic_complexity", features['lowlevel.dynamic_complexity'])
            # print()
            # print("info de :lowlevel.erbbands.dmean", len(features['lowlevel.erbbands.dmean']))
            # print("info de :lowlevel.erbbands.dmean2", len(features['lowlevel.erbbands.dmean2']))
            # print("info de :lowlevel.erbbands.dvar", len(features['lowlevel.erbbands.dvar']))
            # print("info de :lowlevel.erbbands.dvar2", len(features['lowlevel.erbbands.dvar2']))
            # print("info de :lowlevel.erbbands.max", len(features['lowlevel.erbbands.max']))
            # print("info de :lowlevel.erbbands.mean", len(features['lowlevel.erbbands.mean']))
            # print("info de :lowlevel.erbbands.median", len(features['lowlevel.erbbands.median']))
            # print("info de :lowlevel.erbbands.min", len(features['lowlevel.erbbands.min']))
            # print("info de :lowlevel.erbbands.stdev", len(features['lowlevel.erbbands.stdev']))
            # print("info de :lowlevel.erbbands.var", len(features['lowlevel.erbbands.var']))
            # print()
            # print("info de :lowlevel.erbbands.mean DIMENSION", len(features['lowlevel.erbbands.mean']))
            # print("info de :lowlevel.erbbands.stdev DIMENSION", len(features['lowlevel.erbbands.stdev']))
            # print()
            #
            # print("info de :lowlevel.erbbands_crest.mean ", features['lowlevel.erbbands_crest.mean'])
            # print("info de :lowlevel.erbbands_crest.stdev", features['lowlevel.erbbands_crest.stdev'])
            #
            # print("info de :lowlevel.erbbands_flatness_db.mean", features['lowlevel.erbbands_flatness_db.mean'])
            # print("info de :lowlevel.erbbands_flatness_db.stdev", features['lowlevel.erbbands_flatness_db.stdev'])
            #
            # print("info de :lowlevel.erbbands_kurtosis.mean", features['lowlevel.erbbands_kurtosis.mean'])
            # print("info de :lowlevel.erbbands_kurtosis.stdev", features['lowlevel.erbbands_kurtosis.stdev'])
            #
            # print("info de :lowlevel.erbbands_skewness.mean'", features['lowlevel.erbbands_skewness.mean'])
            # print("info de :lowlevel.erbbands_skewness.stdev", features['lowlevel.erbbands_skewness.stdev'])
            #
            # print("info de :lowlevel.erbbands_spread.mean", features['lowlevel.erbbands_spread.mean'])
            # print("info de :lowlevel.erbbands_spread.stdev", features['lowlevel.erbbands_spread.stdev'])
            #
            # print("info de :lowlevel.gfcc.cov MULTIDIMENSION", features['lowlevel.gfcc.cov'].shape)
            # print("info de :lowlevel.gfcc.icov MULTIDIMENSION", features['lowlevel.gfcc.icov'].shape)
            # print("info de :lowlevel.gfcc.mean DIMENSION", len(features['lowlevel.gfcc.mean']))
            #
            # print("info de :lowlevel.hfc.mean", features['lowlevel.hfc.mean'])
            # print("info de :lowlevel.hfc.stdev", features['lowlevel.hfc.stdev'])
            #
            # print("info de :lowlevel.loudness_ebu128.integrated", features['lowlevel.loudness_ebu128.integrated'])
            #
            # print("info de :lowlevel.loudness_ebu128.loudness_range", features['lowlevel.loudness_ebu128.loudness_range'])
            #
            # print("info de :lowlevel.loudness_ebu128.momentary.mean", features['lowlevel.loudness_ebu128.momentary.mean'])
            # print("info de :lowlevel.loudness_ebu128.momentary.stdev", features['lowlevel.loudness_ebu128.momentary.stdev'])
            #
            # print("info de :lowlevel.loudness_ebu128.short_term.mean", features['lowlevel.loudness_ebu128.short_term.mean'])
            # print("info de :lowlevel.loudness_ebu128.short_term.stdev", features['lowlevel.loudness_ebu128.short_term.stdev'])
            #
            # print("info de :lowlevel.melbands.dmean", len(features['lowlevel.melbands.dmean']))
            # print("info de :lowlevel.melbands.dmean2", len(features['lowlevel.melbands.dmean2']))
            # print("info de :lowlevel.melbands.dvar", len(features['lowlevel.melbands.dvar']))
            # print("info de :lowlevel.melbands.dvar2", len(features['lowlevel.melbands.dvar2']))
            # print("info de :lowlevel.melbands.max", len(features['lowlevel.melbands.max']))
            # print("info de :lowlevel.melbands.mean", len(features['lowlevel.melbands.mean']))
            # print("info de :lowlevel.melbands.median", len(features['lowlevel.melbands.median']))
            # print("info de :lowlevel.melbands.min", len(features['lowlevel.melbands.min']))
            # print("info de :lowlevel.melbands.stdev", len(features['lowlevel.melbands.stdev']))
            # print("info de :lowlevel.melbands.var", len(features['lowlevel.melbands.var']))
            # print()
            # print("info de :lowlevel.melbands96.dmean", len(features['lowlevel.melbands96.dmean']))
            # print("info de :lowlevel.melbands96.dmean2", len(features['lowlevel.melbands96.dmean2']))
            # print("info de :lowlevel.melbands96.dvar", len(features['lowlevel.melbands96.dvar']))
            # print("info de :lowlevel.melbands96.dvar2", len(features['lowlevel.melbands96.dvar2']))
            # print("info de :lowlevel.melbands96.max", len(features['lowlevel.melbands96.max']))
            # print("info de :lowlevel.melbands96.mean", len(features['lowlevel.melbands96.mean']))
            # print("info de :lowlevel.melbands96.median", len(features['lowlevel.melbands96.median']))
            # print("info de :lowlevel.melbands96.min", len(features['lowlevel.melbands96.min']))
            # print("info de :lowlevel.melbands96.stdev", len(features['lowlevel.melbands96.stdev']))
            # print("info de :lowlevel.melbands96.var", len(features['lowlevel.melbands96.var']))
            # print()
            #
            # print("info de :lowlevel.melbands_crest.mean", features['lowlevel.melbands_crest.mean'])
            # print("info de :lowlevel.melbands_crest.stdev", features['lowlevel.melbands_crest.stdev'])
            #
            # print("info de :lowlevel.melbands_flatness_db.mean", features['lowlevel.melbands_flatness_db.mean'])
            # print("info de :lowlevel.melbands_flatness_db.stdev", features['lowlevel.melbands_flatness_db.stdev'])
            #
            # print("info de :lowlevel.melbands_kurtosis.mean", features['lowlevel.melbands_kurtosis.mean'])
            # print("info de :lowlevel.melbands_kurtosis.stdev", features['lowlevel.melbands_kurtosis.stdev'])
            #
            # print("info de :lowlevel.melbands_skewness.mean", features['lowlevel.melbands_skewness.mean'])
            # print("info de :lowlevel.melbands_skewness.stdev", features['lowlevel.melbands_skewness.stdev'])
            #
            # print("info de :lowlevel.melbands_spread.mean ", features['lowlevel.melbands_spread.mean'])
            # print("info de :lowlevel.melbands_spread.stdev", features['lowlevel.melbands_spread.stdev'])
            #
            # print("info de :lowlevel.mfcc.cov MULTIDIMENSION", features['lowlevel.mfcc.cov'].shape)
            # print("info de :lowlevel.mfcc.icov MULTIDIMENSION", features['lowlevel.mfcc.icov'].shape)
            # print("info de :lowlevel.mfcc.mean DIMENSION", len(features['lowlevel.mfcc.mean']))
            #
            # print("info de :lowlevel.pitch.mean", features['lowlevel.pitch.mean'])
            # print("info de :lowlevel.pitch.stdev", features['lowlevel.pitch.stdev'])
            #
            # print("info de :lowlevel.pitch_instantaneous_confidence.mean", features['lowlevel.pitch_instantaneous_confidence.mean'])
            # print("info de :lowlevel.pitch_instantaneous_confidence.stdev", features['lowlevel.pitch_instantaneous_confidence.stdev'])
            #
            # print("info de :lowlevel.pitch_salience.mean", features['lowlevel.pitch_salience.mean'])
            # print("info de :lowlevel.pitch_salience.stdev", features['lowlevel.pitch_salience.stdev'])
            #
            # print("info de :lowlevel.silence_rate_20dB.mean", features['lowlevel.silence_rate_20dB.mean'])
            # print("info de :lowlevel.silence_rate_20dB.stdev", features['lowlevel.silence_rate_20dB.stdev'])
            #
            # print("info de :lowlevel.silence_rate_30dB.mean", features['lowlevel.silence_rate_30dB.mean'])
            # print("info de :lowlevel.silence_rate_30dB.stdev", features['lowlevel.silence_rate_30dB.stdev'])
            #
            # print("info de :lowlevel.silence_rate_60dB.dmean", features['lowlevel.silence_rate_60dB.dmean'])
            # print("info de :lowlevel.silence_rate_60dB.dmean2",features['lowlevel.silence_rate_60dB.dmean2'])
            # print("info de :lowlevel.silence_rate_60dB.dvar", features['lowlevel.silence_rate_60dB.dvar'])
            # print("info de :lowlevel.silence_rate_60dB.dvar2", features['lowlevel.silence_rate_60dB.dvar2'])
            # print("info de :lowlevel.silence_rate_60dB.max", features['lowlevel.silence_rate_60dB.max'])
            # print("info de :lowlevel.silence_rate_60dB.mean", features['lowlevel.silence_rate_60dB.mean'])
            # print("info de :lowlevel.silence_rate_60dB.median", features['lowlevel.silence_rate_60dB.median'])
            # print("info de :lowlevel.silence_rate_60dB.min", features['lowlevel.silence_rate_60dB.min'])
            # print("info de :lowlevel.silence_rate_60dB.stdev", features['lowlevel.silence_rate_60dB.stdev'])
            # print("info de :lowlevel.silence_rate_60dB.var", features['lowlevel.silence_rate_60dB.var'])
            # print()
            #
            # print("info de :lowlevel.spectral_centroid.mean", features['lowlevel.spectral_centroid.mean'])
            # print("info de :lowlevel.spectral_centroid.stdev", features['lowlevel.spectral_centroid.stdev'])
            #
            # print("info de :lowlevel.spectral_complexity.mean", features['lowlevel.spectral_complexity.mean'])
            # print("info de :lowlevel.spectral_complexity.stdev", features['lowlevel.spectral_complexity.stdev'])
            #
            # print("info de :lowlevel.spectral_contrast_coeffs.dmean", len(features['lowlevel.spectral_contrast_coeffs.dmean']))
            # print("info de :lowlevel.spectral_contrast_coeffs.dmean2", len(features['lowlevel.spectral_contrast_coeffs.dmean2']))
            # print("info de :lowlevel.spectral_contrast_coeffs.dvar", len(features['lowlevel.spectral_contrast_coeffs.dvar']))
            # print("info de :lowlevel.spectral_contrast_coeffs.dvar2", len(features['lowlevel.spectral_contrast_coeffs.dvar2']))
            # print("info de :lowlevel.spectral_contrast_coeffs.max", len(features['lowlevel.spectral_contrast_coeffs.max']))
            # print("info de :lowlevel.spectral_contrast_coeffs.mean", len(features['lowlevel.spectral_contrast_coeffs.mean']))
            # print("info de :lowlevel.spectral_contrast_coeffs.median", len(features['lowlevel.spectral_contrast_coeffs.median']))
            # print("info de :lowlevel.spectral_contrast_coeffs.min", len(features['lowlevel.spectral_contrast_coeffs.min']))
            # print("info de :lowlevel.spectral_contrast_coeffs.stdev", len(features['lowlevel.spectral_contrast_coeffs.stdev']))
            # print("info de :lowlevel.spectral_contrast_coeffs.var", len(features['lowlevel.spectral_contrast_coeffs.var']))
            # print()
            #
            # print("info de :lowlevel.spectral_contrast_valleys.dmean", len(features['lowlevel.spectral_contrast_valleys.dmean']))
            # print("info de :lowlevel.spectral_contrast_valleys.dmean2", len(features['lowlevel.spectral_contrast_valleys.dmean2']))
            # print("info de :lowlevel.spectral_contrast_valleys.dvar", len(features['lowlevel.spectral_contrast_valleys.dvar']))
            # print("info de :lowlevel.spectral_contrast_valleys.dvar2", len(features['lowlevel.spectral_contrast_valleys.dvar2']))
            # print("info de :lowlevel.spectral_contrast_valleys.max", len(features['lowlevel.spectral_contrast_valleys.max']))
            # print("info de :lowlevel.spectral_contrast_valleys.mean", len(features['lowlevel.spectral_contrast_valleys.mean']))
            # print("info de :lowlevel.spectral_contrast_valleys.median", len(features['lowlevel.spectral_contrast_valleys.median']))
            # print("info de :lowlevel.spectral_contrast_valleys.min", len(features['lowlevel.spectral_contrast_valleys.min']))
            # print("info de :lowlevel.spectral_contrast_valleys.stdev", len(features['lowlevel.spectral_contrast_valleys.stdev']))
            # print("info de :lowlevel.spectral_contrast_valleys.var", len(features['lowlevel.spectral_contrast_valleys.var']))
            # print()
            #
            # print("info de :lowlevel.spectral_crest.mean", features['lowlevel.spectral_crest.mean'])
            # print("info de :lowlevel.spectral_crest.stdev", features['lowlevel.spectral_crest.stdev'])
            #
            # print("info de :lowlevel.spectral_decrease.mean", features['lowlevel.spectral_decrease.mean'])
            # print("info de :lowlevel.spectral_decrease.stdev", features['lowlevel.spectral_decrease.stdev'])
            #
            # print("info de :lowlevel.spectral_energy.mean", features['lowlevel.spectral_energy.mean'])
            # print("info de :lowlevel.spectral_energy.stdev", features['lowlevel.spectral_energy.stdev'])
            #
            # print("info de :lowlevel.spectral_energyband_high.mean", features['lowlevel.spectral_energyband_high.mean'])
            # print("info de :lowlevel.spectral_energyband_high.stdev", features['lowlevel.spectral_energyband_high.stdev'])
            #
            # print("info de :lowlevel.spectral_energyband_low.mean", features['lowlevel.spectral_energyband_low.mean'])
            # print("info de :lowlevel.spectral_energyband_low.stdev", features['lowlevel.spectral_energyband_low.stdev'])
            #
            # print("info de :lowlevel.spectral_energyband_middle_high.mean", features['lowlevel.spectral_energyband_middle_high.mean'])
            # print("info de :lowlevel.spectral_energyband_middle_high.stdev", features['lowlevel.spectral_energyband_middle_high.stdev'])
            #
            # print("info de :lowlevel.spectral_energyband_middle_low.mean", features['lowlevel.spectral_energyband_middle_low.mean'])
            # print("info de :lowlevel.spectral_energyband_middle_low.stdev", features['lowlevel.spectral_energyband_middle_low.stdev'])
            #
            # print("info de :lowlevel.spectral_entropy.mean", features['lowlevel.spectral_entropy.mean'])
            # print("info de :lowlevel.spectral_entropy.stdev", features['lowlevel.spectral_entropy.stdev'])
            #
            #
            # print("info de :lowlevel.spectral_flatness_db.mean", features['lowlevel.spectral_flatness_db.mean'])
            # print("info de :lowlevel.spectral_flatness_db.stdev", features['lowlevel.spectral_flatness_db.stdev'])
            #
            # print("info de :lowlevel.spectral_flux.mean", features['lowlevel.spectral_flux.mean'])
            # print("info de :lowlevel.spectral_flux.stdev", features['lowlevel.spectral_flux.stdev'])
            #
            # print("info de :lowlevel.spectral_kurtosis.mean", features['lowlevel.spectral_kurtosis.mean'])
            # print("info de :lowlevel.spectral_kurtosis.stdev", features['lowlevel.spectral_kurtosis.stdev'])
            #
            # print("info de :lowlevel.spectral_rms.mean", features['lowlevel.spectral_rms.mean'])
            # print("info de :lowlevel.spectral_rms.stdev", features['lowlevel.spectral_rms.stdev'])
            #
            # print("info de :lowlevel.spectral_rolloff.mean", features['lowlevel.spectral_rolloff.mean'])
            # print("info de :lowlevel.spectral_rolloff.stdev ", features['lowlevel.spectral_rolloff.stdev'])
            #
            # print("info de :lowlevel.spectral_skewness.mean", features['lowlevel.spectral_skewness.mean'])
            # print("info de :lowlevel.spectral_skewness.stdev", features['lowlevel.spectral_skewness.stdev'])
            #
            # print("info de :lowlevel.spectral_spread.mean", features['lowlevel.spectral_spread.mean'])
            # print("info de :lowlevel.spectral_spread.stdev", features['lowlevel.spectral_spread.stdev'])
            #
            # print("info de :lowlevel.spectral_strongpeak.mean", features['lowlevel.spectral_strongpeak.mean'])
            # print("info de :lowlevel.spectral_strongpeak.stdev", features['lowlevel.spectral_strongpeak.stdev'])
            #
            # print("info de :lowlevel.zerocrossingrate.mean", features['lowlevel.zerocrossingrate.mean'])
            # print("info de :lowlevel.zerocrossingrate.stdev", features['lowlevel.zerocrossingrate.stdev'])
            # print()
            # # metadata
            # print()
            # print()
            # print()
            # # print("info de :metadata.audio_properties.analysis.downmix", features['metadata.audio_properties.analysis.downmix'])
            # # print("info de :metadata.audio_properties.analysis.equal_loudness", features['metadata.audio_properties.analysis.equal_loudness'])
            # print("info de :metadata.audio_properties.analysis.length", features['metadata.audio_properties.analysis.length'])
            # print("info de :metadata.audio_properties.analysis.sample_rate", features['metadata.audio_properties.analysis.sample_rate'])
            # print("info de :metadata.audio_properties.analysis.start_time'", features['metadata.audio_properties.analysis.start_time'])
            # print("info de :metadata.audio_properties.bit_rate", features['metadata.audio_properties.bit_rate'])
            # # print("info de :metadata.audio_properties.codec", features['metadata.audio_properties.codec'])
            # print("info de :metadata.audio_properties.length", features['metadata.audio_properties.length'])
            # print("info de :metadata.audio_properties.lossless", features['metadata.audio_properties.lossless'])
            # # print("info de :metadata.audio_properties.md5_encoded", features['metadata.audio_properties.md5_encoded'])
            # print("info de :metadata.audio_properties.number_channels", features['metadata.audio_properties.number_channels'])
            # # print("info de :metadata.audio_properties.replay_gain", features['metadata.audio_properties.replay_gain'])
            # print("info de :metadata.audio_properties.sample_rate", features['metadata.audio_properties.sample_rate'])
            # # print("info de :metadata.tags.file_name", features['metadata.tags.file_name'])
            # # print("info de :metadata.version.essentia", features['metadata.version.essentia'])
            # # print("info de :metadata.version.essentia_git_sha", features['metadata.version.essentia_git_sha'])
            # # print("info de :metadata.version.extractor", features['metadata.version.extractor'])
            # print()
            # # rythm
            # print()
            # print("info de :rhythm.beats_count", features['rhythm.beats_count'])
            # print("info de :rhythm.beats_loudness.mean", features['rhythm.beats_loudness.mean'])
            # print("info de :rhythm.beats_loudness.stdev", features['rhythm.beats_loudness.stdev'])
            # print("info de :rhythm.beats_loudness_band_ratio.mean DIMENSION", len(features['rhythm.beats_loudness_band_ratio.mean']))
            # print("info de :rhythm.beats_loudness_band_ratio.stdev DIMENSION", len(features['rhythm.beats_loudness_band_ratio.stdev']))
            # print("info de :rhythm.beats_position DIMENSION", len(features['rhythm.beats_position']))
            # print("info de :rhythm.bpm", features['rhythm.bpm'])
            # print("info de :rhythm.bpm_histogram DIMENSION", len(features['rhythm.bpm_histogram']))
            # print("info de :rhythm.bpm_histogram_first_peak_bpm", features['rhythm.bpm_histogram_first_peak_bpm'])
            # print("info de :rhythm.bpm_histogram_first_peak_weight", features['rhythm.bpm_histogram_first_peak_weight'])
            # print("info de :rhythm.bpm_histogram_second_peak_bpm", features['rhythm.bpm_histogram_second_peak_bpm'])
            # print("info de :rhythm.bpm_histogram_second_peak_spread", features['rhythm.bpm_histogram_second_peak_spread'])
            # print("info de :rhythm.bpm_histogram_second_peak_weight", features['rhythm.bpm_histogram_second_peak_weight'])
            # # print("info de :rhythm.danceability", features['rhythm.danceability'])
            # print("info de :rhythm.onset_rate", features['rhythm.onset_rate'])
            # print()
            # print("info de :'rhythm.bpm_intervals' DIM", len(features['rhythm.bpm_intervals']))
            # print("info de :rhythm.onset_count", features['rhythm.onset_count'])
            # print("info de :'rhythm.onset_rate'", features['rhythm.onset_rate'])
            # print("info de :'rhythm.onset_times' DIM", len(features['rhythm.onset_times']))
            # print()
            # print("info de :'sfx.der_av_after_max'", features['sfx.der_av_after_max'])
            # print("info de :'sfx.duration'", features['sfx.duration'])
            # print("info de :'sfx.effective_duration'", features['sfx.effective_duration'])
            # print("info de :'sfx.flatness'", features['sfx.flatness'])
            # print()
            # print("info de :'sfx.flatness'", features['sfx.flatness'])
            # print()
            #
            # print("info de :sfx.inharmonicity.dmean", features['sfx.inharmonicity.dmean'])
            # print("info de :sfx.inharmonicity.dmean2",features['sfx.inharmonicity.dmean2'])
            # print("info de :sfx.inharmonicity.dvar", features['sfx.inharmonicity.dvar'])
            # print("info de :sfx.inharmonicity.dvar2", features['sfx.inharmonicity.dvar2'])
            # print("info de :sfx.inharmonicity.max", features['sfx.inharmonicity.max'])
            # print("info de :sfx.inharmonicity.mean", features['sfx.inharmonicity.mean'])
            # print("info de :sfx.inharmonicity.median", features['sfx.inharmonicity.median'])
            # print("info de :sfx.inharmonicity.min", features['sfx.inharmonicity.min'])
            # print("info de :sfx.inharmonicity.stdev", features['sfx.inharmonicity.stdev'])
            # print("info de :sfx.inharmonicity.var", features['sfx.inharmonicity.var'])
            # print()
            #
            # print("info de :sfx.logattacktime", features['sfx.logattacktime'])
            # print("info de :sfx.max_der_before_max",features['sfx.max_der_before_max'])
            # print("info de :sfx.max_to_total", features['sfx.max_to_total'])
            # print()
            #
            # print("info de :sfx.oddtoevenharmonicenergyratio.dmean", features['sfx.oddtoevenharmonicenergyratio.dmean'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.dmean2",features['sfx.oddtoevenharmonicenergyratio.dmean2'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.dvar", features['sfx.oddtoevenharmonicenergyratio.dvar'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.dvar2", features['sfx.oddtoevenharmonicenergyratio.dvar2'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.max", features['sfx.oddtoevenharmonicenergyratio.max'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.mean", features['sfx.oddtoevenharmonicenergyratio.mean'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.median", features['sfx.oddtoevenharmonicenergyratio.median'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.min", features['sfx.oddtoevenharmonicenergyratio.min'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.stdev", features['sfx.oddtoevenharmonicenergyratio.stdev'])
            # print("info de :sfx.oddtoevenharmonicenergyratio.var", features['sfx.oddtoevenharmonicenergyratio.var'])
            # print()
            #
            # print("info de :'sfx.pitch_after_max_to_before_max_energy_ratio'", features['sfx.pitch_after_max_to_before_max_energy_ratio'])
            # print("info de :'sfx.pitch_centroid'", features['sfx.pitch_centroid'])
            # print("info de :'sfx.pitch_max_to_total'", features['sfx.pitch_max_to_total'])
            # print("info de :'sfx.pitch_min_to_total'", features['sfx.pitch_min_to_total'])
            # print("info de :'sfx.strongdecay'", features['sfx.strongdecay'])
            # print("info de :'sfx.tc_to_total'", features['sfx.tc_to_total'])
            # print("info de :'sfx.temporal_centroid'", features['sfx.temporal_centroid'])
            # print("info de :'sfx.temporal_decrease'", features['sfx.temporal_decrease'])
            # print("info de :'sfx.temporal_kurtosis'", features['sfx.temporal_kurtosis'])
            # print("info de :'sfx.temporal_skewness'", features['sfx.temporal_skewness'])
            # print("info de :'sfx.temporal_spread'", features['sfx.temporal_spread'])
            # print()
            #
            # print("info de :sfx.tristimulus.dmean", len(features['sfx.tristimulus.dmean']))
            # print("info de :sfx.tristimulus.dmean2",len(features['sfx.tristimulus.dmean2']))
            # print("info de :sfx.tristimulus.dvar", len(features['sfx.tristimulus.dvar']))
            # print("info de :sfx.tristimulus.dvar2", len(features['sfx.tristimulus.dvar2']))
            # print("info de :sfx.tristimulus.max", len(features['sfx.tristimulus.max']))
            # print("info de :sfx.tristimulus.mean", len(features['sfx.tristimulus.mean']))
            # print("info de :sfx.tristimulus.median", len(features['sfx.tristimulus.median']))
            # print("info de :sfx.tristimulus.min", len(features['sfx.tristimulus.min']))
            # print("info de :sfx.tristimulus.stdev", len(features['sfx.tristimulus.stdev']))
            # print("info de :sfx.tristimulus.var", len(features['sfx.tristimulus.var']))
            # print()
            #
            #
            # # tonal
            # print()
            # print("info de :tonal.chords_changes_rate", features['tonal.chords_changes_rate'])
            # print("info de :tonal.chords_histogram DIMENSION", len(features['tonal.chords_histogram']))
            # print("info de :tonal.chords_key", features['tonal.chords_key'])
            # print("info de :tonal.chords_number_rate", features['tonal.chords_number_rate'])
            # print("info de :tonal.chords_scale", features['tonal.chords_scale'])
            # print("info de :tonal.chords_strength.mean", features['tonal.chords_strength.mean'])
            # print("info de :tonal.chords_strength.stdev", features['tonal.chords_strength.stdev'])
            # print("info de :tonal.hpcp.mean DIMENSION", len(features['tonal.hpcp.mean']))
            # print("info de :tonal.hpcp.stdev DIMENSION", len(features['tonal.hpcp.stdev']))
            # print("info de :tonal.hpcp_crest.mean", features['tonal.hpcp_crest.mean'])
            # print("info de :tonal.hpcp_crest.stdev", features['tonal.hpcp_crest.stdev'])
            # print("info de :tonal.hpcp_entropy.mean", features['tonal.hpcp_entropy.mean'])
            # print("info de :tonal.hpcp_entropy.stdev", features['tonal.hpcp_entropy.stdev'])
            # print("info de :tonal.key_edma.key", features['tonal.key_edma.key'])
            # print("info de :tonal.key_edma.scale", features['tonal.key_edma.scale'])
            # print("info de :tonal.key_edma.strength", features['tonal.key_edma.strength'])
            # print("info de :tonal.key_krumhansl.key", features['tonal.key_krumhansl.key'])
            # print("info de :tonal.key_krumhansl.scale", features['tonal.key_krumhansl.scale'])
            # print("info de :tonal.key_krumhansl.strength", features['tonal.key_krumhansl.strength'])
            # print("info de :tonal.key_temperley.key", features['tonal.key_temperley.key'])
            # print("info de :tonal.key_temperley.scale", features['tonal.key_temperley.scale'])
            # print("info de :tonal.key_temperley.strength", features['tonal.key_temperley.strength'])
            # # print("info de :tonal.thpcp", features['tonal.thpcp'])
            # # print("info de :tonal.tuning_diatonic_strength", features['tonal.tuning_diatonic_strength'])
            # # print("info de :tonal.tuning_equal_tempered_deviation", features['tonal.tuning_equal_tempered_deviation'])
            # # print("info de :tonal.tuning_frequency", features['tonal.tuning_frequency'])
            # # print("info de :tonal.tuning_nontempered_energy_ratio", features['tonal.tuning_nontempered_energy_ratio'])
            # print()
            # print("info de :'tonal.chords_count'", features['tonal.chords_count'])
            # print("info de :'tonal.chords_progression' DIM", len(features['tonal.chords_progression']))
            # print("info de :'tonal.hpcp_peak_count'", features['tonal.hpcp_peak_count'])
            # print()
            # print("info de :tonal.tuning_frequency.dmean", features['tonal.tuning_frequency.dmean'])
            # print("info de :tonal.tuning_frequency.dmean2",features['tonal.tuning_frequency.dmean2'])
            # print("info de :tonal.tuning_frequency.dvar", features['tonal.tuning_frequency.dvar'])
            # print("info de :tonal.tuning_frequency.dvar2", features['tonal.tuning_frequency.dvar2'])
            # print("info de :tonal.tuning_frequency.max", features['tonal.tuning_frequency.max'])
            # print("info de :tonal.tuning_frequency.mean", features['tonal.tuning_frequency.mean'])
            # print("info de :tonal.tuning_frequency.median", features['tonal.tuning_frequency.median'])
            # print("info de :tonal.tuning_frequency.min", features['tonal.tuning_frequency.min'])
            # print("info de :tonal.tuning_frequency.stdev", features['tonal.tuning_frequency.stdev'])
            # print("info de :tonal.tuning_frequency.var", features['tonal.tuning_frequency.var'])
            # print()

            import json
            import csv
            import subprocess
            from pandas.io.json import json_normalize
            # with open(file_json) as tmpjson:
            #     jsontmpdata = json.load(tmpjson)
            #     print(len(jsontmpdata))
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
            array_features_size.append(len(features.split(',')))
            content = input_file.readline()
        print(display_ui)
        print("file processed %d / %d" % (cpt_file,nb_files))
        print(display_ui)
        print("features size :")
        print(array_features_size)
        print(display_ui)
out_file.close()
