#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask application for the Bandcamp label classificator / admin interface

"""

from flask import Flask, render_template, request, send_from_directory, jsonify, abort
import logging
import pandas as pd
from clusterer import process_label
from sqlalchemy import create_engine
import psycopg2
import os
import requests
import re
from waitress import serve


app = Flask(__name__)


@app.route('/index')
@app.route('/')
def index():                                                            # main page for clusterisation
    global labels
    pguser = os.environ.get('POSTGRES_USER')
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres')  # pg connect    
    if pg.dialect.has_table(pg, 'labels'):
        labeltable = pd.read_sql_table('labels', pg)                    # check available labels
        pg.dispose()
        labels = labeltable[(labeltable['numtracks'] > 99) & (labeltable['ready'])]['bandcampname'].tolist()
        labels.sort()
        return render_template("index.html", labels = labels)           # render index.html with selectable list of labels
    else:
        pg.dispose()
        return render_template("index.html", labels = [])               # nothing to classify yet



@app.route('/warning_message', methods=['POST'])
def warning_message():
    global processing
    if processing == True:
        return jsonify(message='Processing, please wait!')
    else:
        return jsonify(message='')


@app.route('/classifier', methods=['GET'])
def classifier():
    global processing
    processing = True
    global labels
    usrinput = dict(request.args)
    if 'labels' in usrinput:
        labeltitle = usrinput['labels']
        clusternum = usrinput['clusternum']
        if clusternum == 'Auto':
            clusternum = 0
            logging.info(f'Distributing {labeltitle} into automatically determined number of clusters...')
        else:
            logging.info(f'Distributing {labeltitle} into {clusternum} clusters...')
        process_label(labeltitle, clusternum)
        processing = False
        try:
            return send_from_directory(directory='../config/labels',filename=f'{labeltitle}{clusternum}.pdf',mimetype='application/pdf', as_attachment=True)
        except FileNotFoundError:
            abort(404)
    else:
        return render_template("index.html", labels = [])               # nothing to classify yet
            

@app.route('/addlibrary', methods=['GET'])                              # add new library for processing (admin interface)
def addlibrary():
    usrinput = dict(request.args)
    if len(usrinput['libname']) > 0:
        libname = usrinput['libname']
        urlrequest = requests.get(f'https://{libname}.bandcamp.com/music')
        try:
            regex_title = r'<meta property="og:site_name" content=".*">'
            labelname = re.findall(regex_title, urlrequest.text)[0]     # this will generate an exception if the url does not point to a bandcamp music page
            labelname = labelname[39:-2]
            configdata=pd.read_csv('../config/config.csv')              # import sraping info from config file
            if configdata['scraping'].values[0] > 0:                    # if another scraping/analysis is underway, try to estimate the number of files left
                labeldata=pd.read_csv('../config/labels/currently_scraping/label.csv')
                if configdata['scraping'].values[0] == 2:
                    filesleftnum = str(len([x for x in os.listdir('../config/labels/currently_scraping') if (x.endswith('.mp3'))]))
                else:
                    filesleftnum = "unknown"
                return render_template("admin.html", warning_message = f"Submission not possible at this time.. currently analysing the Bandcamp releases of {labeldata['label_name'].values[0]}. Number of files left: {filesleftnum}, please try again later.", disabled = "disabled")
            else:
                configdata['bclabel'].values[0] = libname               # set pipeline config file to start processing the files
                configdata.to_csv('../config/config.csv', index=False)
                return render_template("admin.html", warning_message = f"Success! The analysis of {labelname} tracks on Bandcamp is underway.\nThis may take several (or many) hours. Number of files left: unknown yet, please check back later.", disabled = "disabled")
        except:
            return render_template("admin.html", warning_message = f"https://{libname}.bandcamp.com is not a valid bandcamp label/library page.")
    else:
        return render_template("admin.html", warning_message = "Please enter a valid bandcamp label/library page.")        


@app.route('/admin', methods=['GET'])                                   # admin page: needs to show if new label submission is possible or not (when another label is being processed already)
def admin():
    configdata=pd.read_csv('../config/config.csv')                      # import sraping info from config file
    if configdata['scraping'].values[0] > 0:
        labeldata=pd.read_csv('../config/labels/currently_scraping/label.csv')
        if configdata['scraping'].values[0] == 2:                       # estimate number of files left for processing:
            filesleftnum = str(len([x for x in os.listdir('../config/labels/currently_scraping') if (x.endswith('.mp3'))]))
        else:
            filesleftnum = "unknown yet"
        return render_template('admin.html', warning_message = f"Submission not possible at this time.. currently analysing the Bandcamp releases of {labeldata['label_name'].values[0]}. Number of files left: {filesleftnum}, please try again later.", disabled = "disabled")
    else:
        return render_template('admin.html', warning_message = "", disabled = "")


@app.route('/gettracks', methods=['GET'])                               # get tracks table (admin interface)
def gettracks():
    pguser = os.environ.get('POSTGRES_USER')                            # setup postgres connection
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres')
    if pg.dialect.has_table(pg, 'tracks'):
        tracks = pd.read_sql_table('tracks', pg, columns=['trackid','artist','year','track','album','tags','labelid','labelname','url','bclabel'])
        pg.dispose()
        tracks.to_csv('tracks.csv', index=False)                        # fetch tracks table
        return send_from_directory(directory='.',filename='tracks.csv',mimetype='text/csv', as_attachment=True)
    else:
        pg.dispose()
        return render_template('admin.html', warning_message = "Tracks database not found.", disabled = "")        


@app.route('/getlabels', methods=['GET'])                               # get label shortlist (admin interface)
def getlabels():
    pguser = os.environ.get('POSTGRES_USER')
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres')  # pg connect    
    if pg.dialect.has_table(pg, 'labels'):
        labeltable = pd.read_sql_table('labels', pg)
        pg.dispose()
        labeltable.to_csv('labels.csv', index=False)                    # fetch labels table
        return send_from_directory(directory='.',filename='labels.csv',mimetype='text/csv', as_attachment=True)
    else:
        pg.dispose()
        return render_template('admin.html', warning_message = "No Bandcamp labels/libraries analysed yet", disabled = "")

if __name__ == '__main__':
    global processing
    processing = False
    logging.basicConfig(level=logging.INFO)    
    app.config['SECRET_KEY'] = os.environ.get('FLASK_KEY')
    serve(app, host="0.0.0.0", threads=10, port=8080)