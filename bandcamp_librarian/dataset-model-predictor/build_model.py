#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Builds classification model using the Beatport 2021 Jan dataset
Follows the methodology of Caparrini et al. (https://doi.org/10.1080/09298215.2020.1761399)
Uses 92 features extracted with pyAudioAnalysis and Essentia

"""

import pickle
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score


print('Generating model..')
df = pd.read_csv('beatport_2021jan.csv')
X = df.drop(['genre','artist','track'], axis=1)
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

classweight = {"Afro House": 1,"Ambient": 0.5,"Bass House": 1,"Big Room": 1,"Breaks": 1,"Dance-Electro Pop": 1,"Deep House": 1,"Drum&Bass": 1,"Dubstep": 1,"Electro (Classic-Detroit-Modern)": 1,"Electro House": 1,"Funky-Groove-Jackin' House": 1,"Future House": 1,"Garage-Bassline-Grime": 1,"Hard Dance-Hardcore": 1,"Hard Techno": 1,"House": 1,"Indie Dance": 1,"Leftfield Bass": 1,"Leftfield House & Techno": 1,"Melodic House&Techno": 1,"Minimal-Deep Tech": 1,"Nu Disco-Disco": 1,"Organic House-Downtempo": 1,"Progressive House": 1,"Psy-Trance": 1,"Reggae-Dancehall-Dub": 1,"Tech House": 1,"Techno (Peak Time-Driving)": 1,"Techno (Raw-Deep-Hypnotic)": 1,"Trance": 1,"Trap-Hip-Hop-R&B": 1}    
BP2021 = ExtraTreesClassifier(bootstrap=False, class_weight=classweight, criterion='gini',
           max_depth=None, max_features=40,
           max_leaf_nodes=None, min_impurity_decrease=0,
           min_samples_leaf=1, min_samples_split=12,
           min_weight_fraction_leaf=0, n_estimators=125, n_jobs=4,
           oob_score=False, random_state=42, verbose=0, warm_start=False)

accuracy = cross_val_score(BP2021, X_train, y_train, cv=10)
print(f'Train CV (10-fold) accuracy: {round(accuracy.mean(),3)} +/- {round(accuracy.std(), 3)}')
BP2021.fit(X_train, y_train)
print(f'Test F1 score: {round(f1_score(y_test, BP2021.predict(X_test), average="macro"), 3)}')
print('Dumping classifier into the /config folder.')
pickle.dump(BP2021, open('../config/beatport_classifier.sav', 'wb'))