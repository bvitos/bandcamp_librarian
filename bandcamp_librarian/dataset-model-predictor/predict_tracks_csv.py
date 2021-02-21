#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Predicts musical subgenres of electronic dance music tracks in CSV file received through command-line argument
Follows the methodology of Caparrini et al. (https://doi.org/10.1080/09298215.2020.1761399)
Uses 92 features extracted with pyAudioAnalysis and Essentia
These features should be located in the first 92 columns of the CSV in the default order of extraction

"""

import pickle
import pandas as pd
import argparse
import os.path

parser = argparse.ArgumentParser(description='Predicts musical genre of electronic dance music tracks based on 92 musical features, using a model trained on Beatport Top100 lists (cf. Caparrini et al. 2020)')
parser.add_argument('-f', '--file', help='CSV file with 92 pyAudioAnalysis and Essentia features', type = str)

args = parser.parse_args()
csv_file = args.file

if csv_file == None:
    print ('Please provde the list of track featuers in a CSV file. Command line argument: -f {filename} or --file {filename}')
else:
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        clf = pickle.load(open("../config/beatport_classifier.sav", 'rb'))
        features_table = df.iloc[:,:92]
        df = df.assign(Predicted_Genre=pd.Series(clf.predict(features_table.to_numpy())))
        resultsfile = csv_file[:-4] + '_predictions' + csv_file[-4:]
        df.to_csv(resultsfile, index=False)
        print(f'Results saved into: {resultsfile}')
    else:
        print(f'File not found: {csv_file}')