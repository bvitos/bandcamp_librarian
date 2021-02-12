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
from sklearn.model_selection import train_test_split


print('Generating model..')
df = pd.read_csv('beatport_2021jan.csv')
df.drop(['artist','track'], axis=1, inplace=True) 
X = df.drop('genre', axis=1)
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

BP2021 = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=0.77,
           max_leaf_nodes=None, min_impurity_decrease=0,
           min_samples_leaf=1, min_samples_split=8,
           min_weight_fraction_leaf=0, n_estimators=325, n_jobs=4,
           oob_score=False, random_state=42, verbose=0, warm_start=False)
BP2021.fit(X_train, y_train)
print('Done! Dumping classifier into the /config folder.')
pickle.dump(BP2021, open('../config/beatport_classifier.sav', 'wb'))