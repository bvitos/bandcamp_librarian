#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import the tracks table from a CSV file
"""

from sqlalchemy import create_engine
import pandas as pd
import psycopg2
import os

if os.path.exists('tracks.csv'):
    print("Please note that you need a running instance of the docker container for importing the tracks and labels tables to Postgres")
    pguser = input("User name: ")
    pgpassword = input("Password: ")
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@0.0.0.0:5555/postgres') # pg connect
    tracksimport = pd.read_csv('tracks.csv', delimiter = ',', decimal = '.')
    tracksimport.iloc[:,:91] = tracksimport.iloc[:,:91].apply(pd.to_numeric, errors='coerce', downcast='float')
    tracksimport[['trackid', 'year', 'labelid']] = tracksimport[['trackid', 'year', 'labelid']].apply(pd.to_numeric, errors='coerce', downcast='integer')
    tracksimport.to_sql('tracks', pg, if_exists='replace', method='multi', index = False, chunksize=1000)  # import to postgres
    print("Postgres table imported.")
else:
    print("Error: could not find tracks.csv")
if os.path.exists('labels/labels.csv'):
    labelsimport = pd.read_csv('labels/labels.csv', delimiter = ',', decimal = '.')
    labelsimport[['labelid', 'numtracks']] = labelsimport[['labelid', 'numtracks']].apply(pd.to_numeric, errors='coerce', downcast='integer')
    labelsimport.to_sql('labels', pg, if_exists='replace', method='multi', index = False, chunksize=1000)  # import to postgres
else:
    print("Error: could not find labels/labels.csv")
pg.dispose()