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
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
from fpdf import FPDF
from scipy.spatial.distance import cdist
from matplotlib.ticker import FuncFormatter
from sqlalchemy import create_engine
from kneed import KneeLocator
import os
import gc
import logging
import base64
from random import randrange



def clusterise_label(dframe, labelname, n_clusters, randomstring):
    """
    Distribute label stored in input dataframe into n clusters

    Returns:
        array of strings used for generating the results PDF file

    """
    clf = pickle.load(open('../config/beatport_classifier.sav', 'rb')) 
    
    df = dframe[dframe['bclabel']==labelname].copy(deep=True)
    features_table = df.iloc[:,:92].copy(deep=True)
    proba_features=pd.DataFrame(data=clf.predict_proba(features_table.to_numpy()),columns=clf.classes_)          # store predicted features df = subgenre probabilities for each track

    configdata=pd.read_csv('../config/config.csv')
    prediction_weight = configdata['prediction_weight'].values[0]   # if the value is 2, the cluster/genre prediction will be aimed towards purer subgenres; if it is 0.5, it will be aimed towards a better amalgam of subgenres

    attempts = 1
    while attempts > 0:                                             # attempts will be set to 0 once elbow point is found during automatic cluster number detection OR if cluster number is user-defined

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
                    element = clustercenters.tolist()[j] # element = the centroid probability values of each cluster
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

    clusters_pred = Kmean.predict(proba_features)                   # clusters_pred would contain the dataframe column with the predicted cluster indexes
    cluster_sizes=[]                                                # but we'll resort the cluster indexes according to the number of samples they contain
    for i in range(n_clusters):                                     # so Cluster 0 will be the largest group, Cluster 1 the second largest, etc.
        cluster_sizes.append(clusters_pred.tolist().count(i))       # first store the cluster sizes
    new_cluster_order = np.argsort(cluster_sizes)[::-1].tolist()    # then get the new order of cluster indexes (sorted according to cluster sizes)
    
    clusters_resorted=[]
    for clust in clusters_pred:
        clusters_resorted.append(new_cluster_order.index(clust))  # generate a new list with the resorted indexes
    df['Cluster']=np.array(clusters_resorted)                       # this will be the Cluster column of the original df
    proba_features['Cluster']=np.array(clusters_resorted)                 # as well as the proba_features df
    clustertext = []
    clustercenters = Kmean.cluster_centers_  # get the actual cluster centers
    
    for counter in range(n_clusters):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

        element = clustercenters.tolist()[new_cluster_order.index(counter)] # element = the centroid probability values of each cluster
        pred_list = list(zip(clf.classes_, element))
        pred_list.sort(key = lambda x: x[1], reverse=True)          # sort it in order of probability, the subgenres with the three highest values will be stored:
        cluster_df = df[df['Cluster']==counter].copy(deep=True)     # store the dataframe cluster in a separate view for easier reference
        cluster_features = proba_features[proba_features['Cluster']==counter].copy(deep=True)     # store the subgenre probas dataframe cluster in a separate df view for easier reference
        clustertext.append(f'Cluster {counter}: elements of {pred_list[0][0]} ({round(pred_list[0][1],2)}), {pred_list[1][0]} ({round(pred_list[1][1],2)}) and {pred_list[2][0]} ({round(pred_list[2][1],2)}).')
        clustertext.append(f"This cluster includes {cluster_df.shape[0]} tracks. Examples (clicking on the cover will take you to the track's Bandcamp page):")
        distances = cdist(cluster_features.iloc[:,:32], np.array(element).reshape(1,-1), 'euclidean').flatten().tolist() # calculate the distances from subgenre probabilities centroid

        asc_index = np.argsort(distances).tolist()                  # get the indexes of sorted distances from centroid
        found_duplicate = True
        while (len(asc_index) > 3) & (found_duplicate):             # try to select three tracks from different artists
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
        while j < 3 and len(asc_index) > j:                         # stores the three most fitting (closest to their centroid) track examples (provided that they exist) with links and release covers
            clustertext.append(f'Link: {cluster_df.iloc[asc_index[j],100]}')
            clustertext.append(cluster_df.iloc[asc_index[j],101])
            imgstr = cluster_df.iloc[asc_index[j],103]
            if imgstr != '../config/_nocover_.jpg':
                bas64bytes = base64.b64decode(imgstr.encode())              
                filename = f'../config/labels/{randomstring}{cluster_df.iloc[asc_index[j],92]}{str(counter * 3 + j)}.jpg'
                fh = open(filename, 'wb')
                fh.write(bas64bytes)
                fh.close()
                clustertext.append(filename)
            else:
                clustertext.append(imgstr)
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



def process_label(labelname, clusternum):
    """
    Clusterise/classify a Bandcamp library

    Returns:
        the function does not return anything
    Instead, it generates a PDF file containing the classification results (clusters, subgenres, Bandcamp track examples incl. covers, tags/folksonomies and links)

    """
    labels = [labelname]                                                # the script currently processes one label/library only, but with an easy modification of the code multiple labels can be processed at the same time
    pguser = os.environ.get('POSTGRES_USER')                            # setup postgres connection
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres')
    cltext = []
    if pg.dialect.has_table(pg, 'tracks'):
        df = pd.read_sql_table('tracks', pg)
        randomstring = str(randrange(100000000,999999999))
        cltext.append(clusterise_label(df, labelname, int(clusternum), randomstring))
    else:
        cltext.append(['Error: Tracks table not found in the database.'])
    pg.dispose()        
    pdf = FPDF()
    for i in range(len(labels)):                                        # generate PDF report
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
                pdf.image(f'../config/labels/{randomstring}{labels[i]}.png', x=20, y=0, w=160)        
                pdf.set_y(120)
                pdf.set_font('DejaVu', size=9)
                pdf.cell(200, 9, txt=text2, ln=1, align="C")
                pdf.set_font('DejaVu', size=9)
                pdf.cell(200, 9, txt=labeltext[j+1], ln=20, align="C")
            elif text2[0:5] == 'Link:':
                top = pdf.y
                pdf.image(labeltext[j+2], x=5, y=pdf.y, w=45, h=45, link = labeltext[j+1])
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