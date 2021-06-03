#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Estimates clusters among tracks of Bandcamp labels, 
based on features extracted with pyAudioAnalysis and Essentia

The classification model is based on the Electronic Music Features - 201802 BeatportTop100 dataset
https://www.kaggle.com/caparrini/electronic-music-features-201802-beatporttop100
See Caparrini et al. 2020 (https://doi.org/10.1080/09298215.2020.1761399)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import pickle
import os
import gc
import logging
import spacy
import requests

#from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator

from PIL import Image
from fpdf import FPDF
from sqlalchemy import create_engine
from random import randrange
from collections import Counter


def clusterise_label(df, labelname, n_clusters, prediction_weight, randomstring):
    """
    Distribute label stored in input dataframe into n clusters

    Returns:
        array of strings used for generating the results PDF file

    """
    clf = pickle.load(open('../config/beatport_classifier.sav', 'rb')) 
    
    features_table = df.iloc[:,:92].copy(deep=True)
    proba_features=pd.DataFrame(data=clf.predict_proba(features_table.to_numpy()),columns=clf.classes_)          # predicted features df -> stores subgenre probabilities for each track

    attempts = 1
    while attempts > 0:                                                 # attempts will be set to 0 once elbow point is found during automatic cluster number detection OR if cluster number is user-defined

        inertia = []
        if n_clusters == 0:                                             # automatic cluster number detection: the loop will calculate 6 inertia values and search for the elbow point of the curve
            K = range(1, 7)
        else:
            K = range(n_clusters, n_clusters + 1)                       # user-defined cluster numbers: the loop will only run once
        Kmean_version = []                                              # stores the K-means models for each version of cluster numbers during automatic detection
        for cluster_nr in K:
            confidence_max = 0
            for i in range(20):                                         # run the K-means algorithm multiple times and store the results with the highest cumulative confidence values of the cluster centres provided by the subgenre classificator
                Kmean_try = MiniBatchKMeans(init="k-means++", n_clusters=cluster_nr) # this way, clusters that are crystallised around the existing taxonomy should be selected // MiniBatchKmeans for faster results
#                Kmean_try = KMeans(init="k-means++", n_clusters=cluster_nr)
                Kmean_try.fit(proba_features)
                confidence_sum = 0
                clustercenters = Kmean_try.cluster_centers_             # store cluster centers
                for j in range(cluster_nr):
                    element = clustercenters.tolist()[j]                # element = the centroid probability values of each cluster
                    pred_list = list(zip(clf.classes_, element))
                    pred_list.sort(key = lambda x: x[1], reverse=True)  # sort them in order of probability
                    confidence_sum += pred_list[0][1] * prediction_weight + pred_list[1][1] + pred_list[2][1] # add together the highest probability values, taking into account the cluster genre prediction weight in order to gain purer subgenres (2x) or a better amalgam of subgenres (0.5x)
                if confidence_sum > confidence_max:                     # store the model with the highest cumulative confidence value
                    confidence_max = confidence_sum
                    Kmean = Kmean_try
            Kmean_version.append(Kmean)
            inertia.append(Kmean.inertia_)

        if n_clusters == 0:                                             # in case of automatic cluster numbers detection: determine the elbow point of the inertia curve, and store it as n_clusters
            kl = KneeLocator(K, inertia, curve="convex", direction="decreasing")
            n_clusters = kl.elbow
            logging.info(f'Number of clusters for {labelname}: {n_clusters}.')
            try:
                Kmean = Kmean_version[n_clusters - 1]                   # if valid elbow point found, store the final model
            except:
                if attempts < 5:                                        # if no elbow point found (n_clusters is None): repeat the loop max. 5 times
                    attempts += 1
                    n_clusters = 0
                else:
                    attempts = 0
                    n_clusters = 4                                      # if elbow point cannot be determined after 5 attempts, just use a default nr of 4 clusters
                    Kmean = Kmean_version[n_clusters - 1]               # store the final model
        else:
            attempts = 0                                                # if cluster number was defined by user, only run the loop once

    clusters_pred = Kmean.predict(proba_features)                       # clusters_pred will be the Cluster column of the main df
    df['Cluster']=np.array(clusters_pred)
    proba_features['Cluster']=np.array(clusters_pred)                   # as well as the proba_features df
    clustertext = []
    clustercenters = Kmean.cluster_centers_                             # get the  cluster centers
    
    for ccounter in range(n_clusters):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

        element = clustercenters.tolist()[ccounter]                     # element = the centroid probability values of each cluster
        pred_list = list(zip(clf.classes_, element))
        pred_list.sort(key = lambda x: x[1], reverse=True)              # sort it in order of probability, the subgenres with the three highest values will be stored:
        cluster_df = df[df['Cluster']==ccounter].copy(deep=True)        # store the dataframe cluster in a separate view for easier reference
        cluster_features = proba_features[proba_features['Cluster']==ccounter].copy(deep=True)     # store the subgenre probas dataframe cluster in a separate df view for easier reference
        clustertext.append(f'Cluster {ccounter}- elements of {pred_list[0][0]} ({round(pred_list[0][1],2)}), {pred_list[1][0]} ({round(pred_list[1][1],2)}) and {pred_list[2][0]} ({round(pred_list[2][1],2)}).')
        locations = []
        for row in cluster_df['GPE']:
            locations = locations + row
        nationalities = []
        for row in cluster_df['NORP']:
            nationalities = nationalities + row
        locationcount = Counter(locations)
        nationalitiescount = Counter(nationalities)
        loctags = 'Locale tags'
        for locelement in sorted(locationcount, key=locationcount.get, reverse=True):
            loctags = loctags + '- ' + locelement + ' (' + str(locationcount[locelement]) + ') '
        for natelement in sorted(nationalitiescount, key=nationalitiescount.get, reverse=True):
            loctags = loctags + '- ' + natelement + ' (' + str(nationalitiescount[natelement]) + ') '
        clustertext.append(loctags)
        clustertext.append(f"The cluster includes {cluster_df.shape[0]} tracks. Notable track examples (clicking on the cover will take you to the Bandcamp release page):")

        distances = cdist(cluster_features.iloc[:,:32], np.array(element).reshape(1,-1), 'euclidean').flatten().tolist() # calculate the distances from subgenre probabilities centroid

        asc_index = np.argsort(distances).tolist()                      # get the indexes of sorted distances from centroid
        found_duplicate = True
        while (len(asc_index) > 3) & (found_duplicate):                 # try to select three tracks from different artists
            found_duplicate = False
            for j in range(3):
                artist_list = [cluster_df.iloc[asc_index[0],93], cluster_df.iloc[asc_index[1],93], cluster_df.iloc[asc_index[2],93]]
                artist_list.pop(j)
                if (cluster_df.iloc[asc_index[j],93] in artist_list) & (cluster_df.iloc[asc_index[j],93] != 'Various') & (cluster_df.iloc[asc_index[j],93] != 'Various Artists'):
                    duplicate = asc_index[j]
                    found_duplicate = True
            if (found_duplicate):
                asc_index.remove(duplicate)
        j = 0
        while j < 3 and len(asc_index) > j:                             # stores the three most fitting (closest to their centroid) track examples (provided that they exist) with links and release covers
            clustertext.append(f'Link: {cluster_df.iloc[asc_index[j],100]}')
            clustertext.append(cluster_df.iloc[asc_index[j],101])
            imgstr = cluster_df.iloc[asc_index[j],103]
            if len(imgstr) > 0:
                try:                                                    # try to download cover image
                    image_data = requests.get(imgstr).content
                    filename = f'../config/labels/{randomstring}{cluster_df.iloc[asc_index[j],92]}{str(ccounter * 3 + j)}.jpg'
                    f = open(filename, 'w+b')
                    f.write(image_data)
                    f.close()
                    img = Image.open(filename)
                    img = img.resize((300,300), Image.ANTIALIAS)
                    img.save(filename)
                    clustertext.append(filename)
                except:                                                 # no cover image available
                    clustertext.append('../config/_nocover_.jpg')
            else:
                clustertext.append('../config/_nocover_.jpg')
            clustertext.append(cluster_df.iloc[asc_index[j],97])
            j += 1
        clustertext.append("Note: the track analysis and consecutive K-means clustering is based on Beatport's taxonomy composed of 32 subgenres as of Jan 2021.")
        clustertext.append("The classifier was trained on the top 100 tracks of each subgenre; the numbers in brackets represent probability/confidence values.")
    fig = plt.figure()
    sns.histplot(data=df, x="year", hue="Cluster", palette = "Set2", element = "poly")
#   sns.kdeplot(data=df, x="Year", hue="Cluster", shade=True, palette = "Set2")             # alternative to histplot
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))                 # only display integers on X axis
    plt.title(df['labelname'].values[0])
    plt.xlabel('Year')
    plt.ylabel('Tracks')
    fig.savefig(f'../config/labels/{randomstring}{labelname}.png', dpi=200)
    plt.close()

    return clustertext



def extract_locales(tags, blacklist=list, GPE_whitelist=list, locale_type=str):
    """
    Extracts locale information from a string containing tags separated by '|' while disregarding blacklisted tags
    spaCy entity types: GPE (Geopolitical entity); NORP (Nationalities or religious or political groups)

    Returns:
        list of locations

    """
    if locale_type+tags in tags_cache:                  # check if the tag/entity pair was already processed; duplicates are common
        return tags_cache[locale_type+tags]
    else:
        results = []
        for tag in tags.split('|'):
            if locale_type == 'GPE' and tag in GPE_whitelist and tag not in results: # add whitelisted location tags instantly
                results.append(tag)
            for ent in nlp(tag.title()).ents:           # check for location info in individual tag (in title case):
                if ent.label_ in [locale_type] and ent.text not in results and ent.text not in blacklist:
                    results.append(ent.text)
        tags_cache[locale_type+tags] = results          # store processed tag/entity string in cache dictionary
        return(results)





def process_label(labelname, clusternum):
    """
    Clusterise/classify a Bandcamp library

    Returns:
        the function does not return anything
    Instead, it generates a PDF file containing the classification results (clusters, subgenres, Bandcamp track examples incl. covers, tags/folksonomies and links)

    """
    labels = [labelname]                                                            # the script currently processes one label/library only, but with an easy modification of the code multiple labels can be processed at the same time
    pguser = os.environ.get('POSTGRES_USER')                                        # setup postgres connection
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgresql://{pguser}:{pgpassword}@pg_container:5432/postgres')
    cltext = []
    if pg.dialect.has_table(pg, 'tracks'):
        randomstring = str(randrange(100000000,999999999))
        df_full = pd.read_sql_table('tracks', pg)
        df = df_full[df_full['bclabel']==labelname].copy(deep=True)
        configdata=pd.read_csv('../config/config.csv')
        loc_tag_blacklist = configdata['loc_tag_blacklist'].values[0].split('|')   # tags incorrectly detected as locales by the NLP algorithm must be blacklisted
        GPE_tag_whitelist = configdata['loc_tag_whitelist'].values[0].split('|')   # locale tags not detected by the NLP algorithm must be whitelisted
        prediction_weight = configdata['prediction_weight'].values[0]              # if the value is 2, the cluster/genre prediction will be aimed towards purer subgenres; if it is 0.5, it will be aimed towards a better amalgam of subgenres
        global nlp
        nlp = spacy.load('en_core_web_md')
        global tags_cache
        tags_cache = {}
        df['GPE'] = df['tags'].apply(extract_locales, args=[loc_tag_blacklist, GPE_tag_whitelist, 'GPE'])       # extract location info into df column
        df['NORP'] = df['tags'].apply(extract_locales, args=[loc_tag_blacklist,GPE_tag_whitelist, 'NORP'])      # extract location info into df column
        cltext.append(clusterise_label(df, labelname, int(clusternum), prediction_weight, randomstring))
    else:
        cltext.append(['Error: Tracks table not found in the database.'])
    pg.dispose()        
    pdf = FPDF()
    for i in range(len(labels)):                                                    # generate PDF report
        pdf.add_page()
        pdf.set_auto_page_break(0, margin = 0.5)
        labeltext = cltext[i]
        labeltext = [x.encode('latin-1', 'replace').decode('latin-1') for x in labeltext]
        labeltext = [x.replace('&amp;', '&') for x in labeltext]
        for j in range(len(labeltext)):
            pdf.add_font('DejaVu', '', '../config/DejaVuSansCondensed.ttf', uni=True)
            pdf.add_font('DejaVuI', '', '../config/DejaVuSansCondensed-Oblique.ttf', uni=True)
            pdf.set_font('DejaVu', size=8)
            text2 = labeltext[j]
            if text2[0:5] == 'Clust':
                if j > 0:
                    pdf.add_page()
                pdf.image(f'../config/labels/{randomstring}{labels[i]}.png', x=30, y=0, w=140)        
                pdf.set_y(110)
                pdf.set_font('DejaVu', size=9)
                pdf.cell(200, 9, txt=text2, ln=1, align="C")
                pdf.set_font('DejaVu', size=9)
                pdf.cell(200, 9, txt=labeltext[j+1], ln=20, align="C")
                pdf.cell(200, 9, txt=labeltext[j+2], ln=20, align="C")
            elif text2[0:5] == 'Link:':
                top = pdf.y
                pdf.image(labeltext[j+2], x=5, y=pdf.y, w=45, h=45, link = labeltext[j+1])
                if labeltext[j+2] != '../config/_nocover_.jpg':
                    os.remove(labeltext[j+2])
                pdf.y = top
                pdf.x = 50
                pdf.set_font("DejaVu", size=7)
                celltext = f"{text2[text2.find('_') + 1:-4]} \n Bandcamp artist tags (folskonomy): \n {labeltext[j+3]}"
                pdf.multi_cell(155, 15, txt=celltext, border=1)
                pdf.y = top + 45
                pdf.x = 0
            elif text2[0:5] == 'Note:':
                pdf.set_font('DejaVuI', size=8)
                pdf.x = 0
                pdf.y = 275
                pdf.cell(200, 8, txt=text2, align="C")
                pdf.x = 0
                pdf.y = 280
                pdf.cell(200, 8, txt=labeltext[j+1], align="C")
            elif text2[0:5] == 'Error':
                pdf.set_font('DejaVuI', size=10)
                pdf.y = 140
                pdf.cell(200, 12, txt=text2, align="C")
    os.remove(f'../config/labels/{randomstring}{labels[i]}.png')
    pdf.output(f'../config/labels/{labelname}{clusternum}.pdf')
    gc.collect()
