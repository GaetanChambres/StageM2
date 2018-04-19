import essentia
import essentia.standard as es

# Compute all features, aggregate only 'mean' and 'stdev' statistics for all low-level, rhythm and tonal frame features
features = es.Extractor('./data/database/DS1/101_1b1_Al_sc_AKGC417L.wav')

# See all feature names in the pool in a sorted order
print(sorted(features.descriptorNames()))
