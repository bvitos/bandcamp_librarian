#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETL script: Processes mp3 files from the scraping directory
(mp3 tag and album cover extraction, conversion to wav, feature extraction)
then stores results in Postgres
"""

from sqlalchemy import create_engine
import time
from mutagen.easyid3 import EasyID3
from extract_features_wav import extract_features_wav_file
import pandas as pd
import logging
import os
import mutagen
import gc
import shutil
from PIL import Image
import base64
import numpy as np


def gettags(filepath, track_id):
    """
    Fetches the ID3 tags from the mp3 file and extracts the cover image
    
    Returns: extracted tags and cover image link
    """
    metadata = mutagen.File(filepath)
    try:
        imgdata = metadata['APIC:Cover'].data
        coverimg = f'../config/labels/currently_scraping/{str(track_id)}.jpg'
        f = open(coverimg, 'w+b')
        f.write(imgdata)
        f.close()
        img = Image.open(coverimg)
        img = img.resize((300,300), Image.ANTIALIAS)
        img.save(coverimg)
        with open(coverimg, 'rb') as imageFile:
            imgstr = base64.b64encode(imageFile.read())
            coverimg=imgstr.decode()
    except:
        coverimg = '../config/_nocover_.jpg'
    try:        
        track = EasyID3(filepath)
    except:
        track = []
    if 'artist' in track:
        artist = track['artist']
    else:
        artist = ['Unknown Artist']
    if 'title' in track:
        track_title = track['title']
    else:
        track_title = ['Unknown Track']
    if 'album' in track:
        album = track['album']
    else:
        album = ['']
    if 'date' in track:
        releaseyear = track['date']
        year = [int(releaseyear[0])]
    else:
        year = pd.array([None], dtype=pd.Int8Dtype())
    if 'genre' in track:
        genre = track['genre']
    else:
        genre = ['']
    if 'website' in track:
        url = track['website']
    else:
        url = ['']
    return coverimg, artist, track_title, album, year, genre, url


time.sleep(5)

pguser = os.environ.get('POSTGRES_USER')                                           # setup postgres credentials
pgpassword = os.environ.get('POSTGRES_PASSWORD')
pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres') # pg connect

logging.basicConfig(level=logging.INFO)

while True:
    time.sleep(5)
    logging.info('Checking if scraping is underway...')
    configdata=pd.read_csv('../config/config.csv')                                  # import pipeline sraping info from config file
    scraping = configdata['scraping'].values[0]
    time.sleep(5)
    #scraping info form config.cfg -> 0: no bandcamp label under scraping or analysis; 1: scraping is underway; 2: scraper container is done with scraping
    if scraping > 0 and pg.dialect.has_table(pg, 'labels'):
        time.sleep(1)
        for file in os.listdir('../config/labels/currently_scraping'):              # first, remove any leftover wav or jpg files from previous scraping
            if file[-4:] == '.wav':
                os.remove(f'../config/labels/currently_scraping/{file}')
            elif file[-4:] == '.jpg':
                os.remove(f'../config/labels/currently_scraping/{file}')
        labeldata = pd.read_csv('../config/labels/currently_scraping/label.csv')      # get label info (label name, bandcamp (url) name, label id)
        label_name = labeldata['label_name'].values[0]
        bandcamp_name = labeldata['bandcamp_name'].values[0]                
        labeltable = pd.read_sql_table('labels', pg)
        label_id = labeltable[labeltable['bandcampname'] == bandcamp_name]['labelid'].values[0]
        track_id = 10000 * label_id + labeltable[labeltable['bandcampname'] == bandcamp_name]['numtracks'].values[0]        # generate id for next track to be analysed
        while scraping > 0:
            if pg.dialect.has_table(pg, 'tracks'):
                params = {'labelid': int(label_id)}                                 # generate a dataframe with the label tracks that are (possibly) in postgres
                label_db = pd.read_sql_query("SELECT * FROM tracks WHERE CAST(labelid AS int) = %(labelid)s;", pg, params = params)
            else:
                label_db = []
            if len(label_db) > 0:                                                   # if pre-existing label:
                label_exists = 1
                if label_db.shape[0] != labeltable[labeltable['labelid'] == label_id]['numtracks'].values[0]: # set correct number of tracks in case of previous error / crash during scraping
                    labeltable.loc[labeltable[labeltable['labelid'] == label_id].index.values[0],'numtracks'] = label_db.shape[0]
                    labeltable.to_sql('labels', pg, if_exists='replace', method='multi', index = False, chunksize=1000)
                    track_id = 10000 * label_id + label_db.shape[0]
                    logging.info(f'Track num readjusted to {label_db.shape[0]}.')
            else:
                label_exists = 0
            files = sorted(os.listdir('../config/labels/currently_scraping'))       # get list of already scraped files, so the two containers can work simultaneously
            data = []
            files_processed = 1     # value will be changed to 0 once a valid file is found for analysis
            for file in files:
                if file[-4:] == '.mp3':
                    filepath = f'../config/labels/currently_scraping/{file}'
                    total, used, free = shutil.disk_usage("../config")
                    if label_exists and file in label_db['filename'].values:        # perhaps file is already in the database:
                        logging.info(f'Skipping file already in the database: {file}')
                        os.remove(filepath)                                         # clean up the mp3 file
                    elif free < 100000000:                                          # perhaps running out of disk space:
                        logging.warning(f'Available space very low (under 100MB). Skipping {file}.')
                        os.remove(filepath)                                         # clean up the mp3 file
                        time.sleep(5)
                    else:
                        logging.info(f'Processing: {file}')
                        files_processed = 0
                        track_id += 1

                        coverimg, artist, track_title, album, year, genre, url = gettags(filepath, track_id)
                        filename = file
                        dftrack = []

                        try:                        
                            os.system(f'ffmpeg -i "{filepath}" -acodec pcm_s16le -ar 22050 -ac 1 -nostats -loglevel 0 -y "{filepath}.wav"') # convert to wav
                            dftrack = extract_features_wav_file(f'{filepath}.wav')   # extract the audio features
                            os.remove(f'{filepath}.wav')                             # clean up the wav file
                        except Exception as e:
                            logging.warning(f'Could not process file: {file}. Error: {str(e)}')
                            track_id -= 1

                        if os.path.exists(filepath):
                            os.remove(filepath)                                         # clean up the mp3 file
                        gc.collect()
                        
                        if len(dftrack) > 0:                                        # update postgres and number of tracks in labels CSV
                            dftrack['trackid'], dftrack['artist'], dftrack['year'], dftrack['track'], dftrack['album'], dftrack['tags'], dftrack['labelid'], dftrack['labelname'], dftrack['filename'], dftrack['url'], dftrack['bclabel'], dftrack['coverimg'] = track_id, artist[0], year[0], track_title[0], album[0], genre[0], int(label_id), label_name, file, url[0], bandcamp_name, coverimg
                            dftrack.iloc[0,:91] = dftrack.iloc[0,:91].apply(pd.to_numeric, downcast='float')
                            dftrack[['trackid', 'labelid']] = dftrack[['trackid', 'labelid']].apply(pd.to_numeric, downcast='integer')
                            dftrack.to_sql('tracks', pg, if_exists='append', method='multi', index = False, chunksize=1000)  # export to postgres
                            labeltable.loc[labeltable[labeltable['labelid'] == label_id].index.values[0],'numtracks'] = track_id % 10000
                            labeltable.to_sql('labels', pg, if_exists='replace', method='multi', index = False, chunksize=1000)

            if files_processed == 1:                                                  # if no valid files found, perhaps the scraping is over
                if scraping == 2:           
                    logging.info(f'Processing of label {label_name} complete!')
                    scraping = 0
                else:       # scraping might be over, but only after doing an extra loop of file search (perhaps another file was downloaded while analysing)
                    logging.info('No new files found in scraping directory, giving it another go...')
                    time.sleep(5)
                    configdata=pd.read_csv('../config/config.csv')                    # check pipeline scraping status
                    scraping = configdata['scraping'].values[0]                       # if the value is 2 (scraping container is finished with job), the loop will run one more time to check if new files were downloaded during the analysis
        labeltable.loc[labeltable[labeltable['labelid'] == label_id].index.values[0],'ready'] = True
        labeltable.to_sql('labels', pg, if_exists='replace', method='multi', index = False, chunksize=1000)        
        configdata['scraping'].values[0] = 0
        configdata.to_csv('../config/config.csv', index=False)                        # inform pipeline that scraping and analysis is over; set info in postgres labels table too