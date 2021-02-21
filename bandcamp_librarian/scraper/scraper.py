#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bandcamp Scraper
Much of this code is borrowed from Rich Jones' SoundScrape
Miserlou / SoundScrape - https://github.com/Miserlou/SoundScrape
"""

import demjson
import html
import os
import re
import requests
import sys
import logging
import time
import pandas as pd
import shutil

from clint.textui import colored, puts, progress
from datetime import datetime
from mutagen.mp3 import MP3, EasyMP3
from mutagen.id3 import APIC, WXXX
from mutagen.id3 import ID3 as OldID3
from subprocess import Popen, PIPE
from os.path import dirname, exists, join
from os import access, mkdir, W_OK

from sqlalchemy import create_engine

if sys.version_info.minor < 4:
    html_unescape = html.parser.HTMLParser().unescape
else:
    html_unescape = html.unescape



####################################################################
# Bandcamp
####################################################################


# Largely borrowed from Ronier's bandcampscrape
def scrape_bandcamp_url(labelid, url, sizelimit, alreadyscraped, num_tracks=sys.maxsize, folders=False, custom_path=''):
    """
    Pull out artist and track info from a Bandcamp URL.

    Returns:
        list: filenames to open
        
    For the current purpose, it scrapes everything into the same folder and attaches a labelid to the flies
    Files will be deleted after feature extractions

    """

    filenames = []
    album_data = get_bandcamp_metadata(url)
    # If it's a list, we're dealing with a list of Album URLs,
    # so we call the scrape_bandcamp_url() method for each one
    if type(album_data) is list:
        for album_url in album_data:
            filenames.append(
                scrape_bandcamp_url(
                    labelid, album_url, sizelimit, alreadyscraped, num_tracks, folders, custom_path
                )
            )
        return filenames

    artist = album_data.get("artist")
    album_name = album_data.get("album_title")

    if folders:
        if album_name:
            directory = artist + " - " + album_name
        else:
            directory = artist
        directory = sanitize_filename(directory)
        directory = join(custom_path, directory)
        if not exists(directory):
            mkdir(directory)

    for i, track in enumerate(album_data["trackinfo"]):

        if i > num_tracks - 1:
            continue

        try:
            total, used, free = shutil.disk_usage("../config")
            tag_tracklink = url[0:url.find("bandcamp.com") + 12] + track["title_link"]            
            if free < 200000000:
                while free < 200000000:
                    logging.warning(f'Running out of space (under 200MB). Trying to download {tag_tracklink} again in a minute.')
                    total, used, free = shutil.disk_usage("../config")
                    time.sleep(60)
            if tag_tracklink in alreadyscraped:
                logging.info(f'Track {tag_tracklink} is already in the database. Skipping.')
            else:

                track_name = track["title"]
                if track["track_num"]:
                    track_number = str(track["track_num"]).zfill(2)
                else:
                    track_number = None
                if track_number and folders:
                    track_filename = '%s - %s.mp3' % (track_number, track_name)
                else:
                    track_filename = '%s.mp3' % (track_name)

                track_filename = sanitize_filename(track_filename)

                if folders:
                    path = join(directory, track_filename)
                else:
                    path = join(custom_path, str(labelid) + '_' + sanitize_filename(artist) + ' - ' + track_filename)

                if exists(path):
                    puts_safe(colored.yellow("Track already downloaded: ") + colored.white(track_name))
                    continue

                if not track['file']:
                    puts_safe(colored.yellow("Track unavailble for scraping: ") + colored.white(track_name))
                    continue

                puts_safe(colored.green("Downloading") + colored.white(": " + track_name))
                path = download_file(track['file']['mp3-128'], path, sizelimit)

                if path == "too_large":
                    logging.info(f'File exceeds the size limit of {sizelimit} bytes. Skipping.')
                    raise Exception('File exceeds size limit.')

                album_year = album_data['album_release_date']
                if album_year:
                    album_year = datetime.strptime(album_year, "%d %b %Y %H:%M:%S GMT").year

                tag_file(path,
                     artist,
                     track_name,
                     album=album_name,
                     year=album_year,
                     genre=album_data['genre'],
                     artwork_url=album_data['artFullsizeUrl'],
                     track_number=track_number,
                     url=tag_tracklink,
                     )

                filenames.append(path)

                logging.info(f'Download complete: {artist} - {track_name}')

        except Exception as e:
            logging.warning("Problem downloading")
            print(e)

    return filenames




def extract_embedded_json_from_attribute(request, attribute, debug=False):
    """
    Extract JSON object embedded in an element's attribute value.

    The JSON is "sloppy". The native python JSON parser often can't deal,
    so we use the more tolerant demjson instead.

    Args:
        request (obj:`requests.Response`): HTTP GET response from which to extract
        attribute (str): name of the attribute holding the desired JSON object
        debug (bool, optional): whether to print debug messages

    Returns:
        The embedded JSON object as a dict, or None if extraction failed
    """
    try:
        embed = request.text.split('{}="'.format(attribute))[1]
        embed = html_unescape(
            embed.split('"')[0]
        )
        output = demjson.decode(embed)
        if debug:
            print(
                'extracted JSON: '
                + demjson.encode(
                    output,
                    compactly=False,
                    indent_amount=2,
                )
            )
    except Exception as e:
        output = None
        if debug:
            print(e)
    return output


def get_bandcamp_metadata(url):
    """
    Read information from Bandcamp embedded JavaScript object notation.
    The method may return a list of URLs (indicating this is probably a "main" page which links to one or more albums),
    or a JSON if we can already parse album/track info from the given url.
    """
    request = requests.get(url)
    output = {}
    try:
        for attr in ['data-tralbum', 'data-embed']:
            output.update(
                extract_embedded_json_from_attribute(
                    request, attr
                )
            )
    # if the JSON parser failed, we should consider it's a "/music" page,
    # so we generate a list of albums/tracks and return it immediately
    except Exception as e:
        regex_all_albums = r'<a href="(.*/(?:album|track)/[^>]+)">'
        all_albums = re.findall(regex_all_albums, request.text, re.MULTILINE)
        album_url_list = list()
        for album in all_albums:
            album_url = album
            if "?" in album:
                album_url = album[0:album.find("?")]
            if album[0:4] != "http":
                album_url = re.sub(r'music/?$', '', url) + album
            album_url_list.append(album_url)
        return album_url_list
    # if the JSON parser was successful, use a regex to get all tags
    # from this album/track, join them and set it as the "genre"
    regex_tags = r'<a class="tag" href[^>]+>([^<]+)</a>'
    tags = re.findall(regex_tags, request.text, re.MULTILINE)
    # make sure we treat integers correctly with join()
    # according to http://stackoverflow.com/a/7323861
    # (very unlikely, but better safe than sorry!)
    output['genre'] = ' '.join(s for s in tags)

    try:
        artUrl = request.text.split("\"tralbumArt\">")[1].split("\">")[0].split("href=\"")[1]
        output['artFullsizeUrl'] = artUrl
    except:
        puts_safe(colored.red("Couldn't get full artwork") + "")
        output['artFullsizeUrl'] = None

    return output



####################################################################
# File Utility
####################################################################


def download_file(url, path, sizelimit, session=None, params=None):
    """
    Download an individual file.
    """

    if url[0:2] == '//':
        url = 'https://' + url[2:]

    # Use a temporary file so that we don't import incomplete files.
    tmp_path = path + '.tmp'

    if session and params:
        r = session.get( url, params=params, stream=True )
    elif session and not params:
        r = session.get( url, stream=True )
    else:
        r = requests.get(url, stream=True)

    logging.info(f'Downloading {int(r.headers.get("content-length", 0))} bytes.')

    if int(r.headers.get('content-length', 0)) < int(sizelimit):
        with open(tmp_path, 'wb') as f:
            total_length = int(r.headers.get('content-length', 0))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        os.rename(tmp_path, path)
        return path
    else:
        return "too_large"


def tag_file(filename, artist, title, year=None, genre=None, artwork_url=None, album=None, track_number=None, url=None, comment=None):
    """
    Attempt to put ID3 tags on a file.

    Args:
        artist (str):
        title (str):
        year (int):
        genre (str):
        artwork_url (str):
        album (str):
        track_number (str):
        filename (str):
        url (str):
    """

    try:
        audio = EasyMP3(filename)
        audio.tags = None
        audio["artist"] = artist
        audio["title"] = title
        if year:
            audio["date"] = str(year)
        if album:
            audio["album"] = album
        if track_number:
            audio["tracknumber"] = track_number
        if genre:
            audio["genre"] = genre
        if url:
            audio["website"] = url
        if comment:
            audio["comment"] = comment
        audio.save()

        if artwork_url:

            artwork_url = artwork_url.replace('https', 'http')

            mime = 'image/jpeg'
            if '.jpg' in artwork_url:
                mime = 'image/jpeg'
            if '.png' in artwork_url:
                mime = 'image/png'

            if '-large' in artwork_url:
                new_artwork_url = artwork_url.replace('-large', '-t500x500')
                try:
                    image_data = requests.get(new_artwork_url).content
                except Exception as e:
                    # No very large image available.
                    image_data = requests.get(artwork_url).content
            else:
                image_data = requests.get(artwork_url).content

            audio = MP3(filename, ID3=OldID3)
            audio.tags.add(
                APIC(
                    encoding=3,  # 3 is for utf-8
                    mime=mime,
                    type=3,  # 3 is for the cover image
                    desc='Cover',
                    data=image_data
                )
            )
            audio.save()

        # because there is software that doesn't seem to use WOAR we save url tag again as WXXX
        if url:
            audio = MP3(filename, ID3=OldID3)
            audio.tags.add( WXXX( encoding=3, url=url ) )
            audio.save()

        return True

    except Exception as e:
        puts(colored.red("Problem tagging file: ") + colored.white("Is this file a WAV?"))
        return False

def open_files(filenames):
    """
    Call the system 'open' command on a file.
    """
    command = ['open'] + filenames
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()


def sanitize_filename(filename):
    """
    Make sure filenames are valid paths.

    Returns:
        str:
    """
    sanitized_filename = re.sub(r'[/\\:*?"<>|]', '-', filename)
    sanitized_filename = sanitized_filename.replace('&', 'and')
    sanitized_filename = sanitized_filename.replace('"', '')
    sanitized_filename = sanitized_filename.replace("'", '')
    sanitized_filename = sanitized_filename.replace("/", '')
    sanitized_filename = sanitized_filename.replace("\\", '')

    # Annoying.
    if sanitized_filename[0] == '.':
        sanitized_filename = u'dot' + sanitized_filename[1:]

    return sanitized_filename

def puts_safe(text):
    if sys.platform == "win32":
        if sys.version_info < (3,0,0):
            puts(text)
        else:
            puts(text.encode(sys.stdout.encoding, errors='replace').decode())
    else:
        puts(text)



####################################################################
# Bandcamp scraper for the label classifier app
####################################################################



def get_label_tracklist(bandcampname,fullname):
    '''
    fetches or generates label ID and updates the labels postgres and CSV table if needed
    '''

    alreadyscraped = []
    pguser = os.environ.get('POSTGRES_USER')
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres')  # pg connect    
    if pg.dialect.has_table(pg, 'labels'):
        
        labeltable = pd.read_sql_table('labels', pg)
        if len(labeltable[labeltable['bandcampname'] == bandcampname]) == 0:            # if new label in list:
            labelid = labeltable['labelid'].max() + 1
            labeltable = labeltable.append(pd.DataFrame({'labelid' : [labelid], 'bandcampname' : [bandcampname], 'labelname' : [fullname], 'numtracks' : [0]}), ignore_index = True)
        else:                                                                           # if pre-existing label:
            labelid = labeltable[labeltable['bandcampname'] == bandcampname]['labelid'].values[0]
            if pg.dialect.has_table(pg, 'tracks'):
                params = {'labelid': int(labelid)}                                      # check label tracks already in postgres
                label_db = pd.read_sql_query("SELECT * FROM tracks WHERE CAST(labelid AS int) = %(labelid)s;", pg, params = params)            
                alreadyscraped = label_db['url'].tolist()
    else:                                                                               # if this is te first label being scraped:
        labelid = 1
        labeltable = pd.DataFrame({'labelid' : [1], 'bandcampname' : [bandcampname], 'labelname' : [fullname], 'numtracks' : [0], 'ready' : [False]})
    
    labeltable['ready'] = True
    labeltable.loc[labeltable[labeltable['labelid'] == labelid].index.values[0],'ready'] = False
    labeltable.to_sql('labels', pg, if_exists='replace', method='multi', index = False, chunksize=1000)  # export label table to postgres too
    pg.dispose()

    return labelid, alreadyscraped



logging.basicConfig(level=logging.INFO)
while True:
    time.sleep(5)
    pguser = os.environ.get('POSTGRES_USER')
    pgpassword = os.environ.get('POSTGRES_PASSWORD')
    pg = create_engine(f'postgres://{pguser}:{pgpassword}@pg_container:5432/postgres')  # pg connect                
    logging.info('Checking if a new label has been assigned for scraping...')
    configdata=pd.read_csv('../config/config.csv')                          # import config settings
    bc_label = configdata['bclabel'].values[0]
    sizelimit = configdata['sizelimit'].values[0]

    if bc_label != '_none_':                                                # if label assigned for scraping:
        logging.info(f'Checking the Bandcamp page of {bc_label}...')
        configdata['scraping'].values[0] = 1
        configdata.to_csv('../config/config.csv', index=False)              # notify pipeline that scraping is underway through config file
        label_url = 'https://' + bc_label + '.bandcamp.com/music'
        request = requests.get(label_url)
        regex_title = r'<meta property="og:site_name" content=".*">'
        labelname = re.findall(regex_title, request.text)[0]
        labelname = labelname[39:-2]                                        # get label name
        labelid, alreadyscraped = get_label_tracklist(bc_label, labelname)  # get label id and list of tracks already scraped
        for file in os.listdir('../config/labels/currently_scraping'):      # remove any leftover mp3 files from scraping folder
            if file[-4:] == '.mp3':
                os.remove(f'../config/labels/currently_scraping/{file}')
        newlabelinfo = pd.DataFrame ({'label_name' : [labelname], 'bandcamp_name' : [bc_label]})
        newlabelinfo.to_csv('../config/labels/currently_scraping/label.csv',index=False)        # add label details to pipeline, to be fetched by the etl container
        bc_url = 'https://' + bc_label + '.bandcamp.com/music'
        filenames = scrape_bandcamp_url(labelid, bc_url, sizelimit, alreadyscraped, custom_path = '../config/labels/currently_scraping/')  # scrape bandcamp files

        configdata['bclabel'].values[0] = "_none_"                          # scraping complete
        configdata['scraping'].values[0] = 2
        configdata.to_csv('../config/config.csv', index=False)              # notify pipeline that scraping is complete
    
    time.sleep(5)