import os
import essentia

# as there are 2 operating modes in essentia which have the same algorithms,
# these latter are dispatched into 2 submodules:
import essentia.standard as std
import essentia.streaming as str

input_directory = './DATASET/DS3/'
output_directory = './DATASET/DS3_Features/'
cpt = 0
nb_files = 305
for filename in os.listdir(input_directory):
    if(filename.endswith('.wav')):
        cpt = cpt+1
        print 'processing file' ,cpt, 'over ',nb_files
        input = input_directory + filename
        #print(input)
        output = output_directory + filename[:-3] + "features"
        #print(output)

        os.system('essentia_streaming_extractor_freesound ' + input + ' ' + output)
