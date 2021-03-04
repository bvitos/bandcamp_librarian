"""Console script for Bandcamp Dance Librarian."""

import pandas as pd
import os
import argparse
import sys
import subprocess
import psycopg2
import time
    
from sqlalchemy import create_engine


def import_tracks(postgres_pwd):
    """
    Imports tracks and labels tables to Postgres database

    """
    if os.path.exists('config/tracks.csv') and os.path.exists('config/labels/labels.csv'):
        print('Importing tables to Postgres database..')
        time.sleep(2)
        connection_error = False
        try:
            pg = create_engine(f'postgres://postgres:{postgres_pwd}@0.0.0.0:5555/postgres')     # try to connect to pg
        except:
            print('Connection failed. Trying again in 10 seconds')
            time.sleep(10)
            try:
                pg = create_engine(f'postgres://postgres:{postgres_pwd}@0.0.0.0:5555/postgres') # try to connect to pg
            except:
                connection_error = True
        if connection_error:
            print('Connection failed. You can try again later using the csvimport.py script in the /config folder.')
        else:
            tracksimport = pd.read_csv('config/tracks.csv', delimiter = ',', decimal = '.')
            tracksimport.iloc[:,:91] = tracksimport.iloc[:,:91].apply(pd.to_numeric, errors='coerce', downcast='float')
            tracksimport[['trackid', 'year', 'labelid']] = tracksimport[['trackid', 'year', 'labelid']].apply(pd.to_numeric, errors='coerce', downcast='integer')
            tracksimport.to_sql('tracks', pg, if_exists='replace', method='multi', index = False, chunksize=1000)  # import to postgres
            labelsimport = pd.read_csv('config/labels/labels.csv', delimiter = ',', decimal = '.')
            labelsimport[['labelid', 'numtracks']] = labelsimport[['labelid', 'numtracks']].apply(pd.to_numeric, errors='coerce', downcast='integer')
            labelsimport.to_sql('labels', pg, if_exists='replace', method='multi', index = False, chunksize=1000)  # import to postgres
            pg.dispose()
            print("Database import complete.")
    else:
        print('Error: could not find one or any of these files: config/tracks.csv; config/labels/labels.csv')
    

def main():
    """
    Console script main function. Options: -on -off

    """    
    if len(sys.argv)==1:    
        print('Bandcamp Dance Librarian console script. Options: -on -off')
    else:
        parser = argparse.ArgumentParser(description='Bandcamp Dance Librarian console script. Options: -on -off')
        parser.add_argument('-on', help='Start the application Docker containers. If no environmental variables file found, it initializes one and builds the containers.', action='store_true')
        parser.add_argument('-off', help='Stops the application Docker containers.', action='store_true')
        args, unknown = parser.parse_known_args()
        if len(unknown) > 0:
            print(f'Undefined: {unknown[0]}')
        elif args.on:
            if args.off:
                print('Cannot use -on and -off at the same time.')
            else:
                os.chdir(os.path.dirname(os.path.realpath(__file__)))
                if os.path.exists('.vars.env'):
                    print('If requested, please enter your root password for starting the Bandcamp Dance Librarian Docker containers in the background.')
                    try:
                        output = subprocess.check_output(['bash','-c', 'sudo docker-compose kill'])
                        output = subprocess.check_output(['bash','-c', 'sudo docker-compose up --detach --force-recreate'])
                    except:
                        output = subprocess.check_output(['bash','-c', 'docker-compose kill'])
                        output = subprocess.check_output(['bash','-c', 'docker-compose up --detach --force-recreate'])     # for AWS etc. deployment
                    print('The web interface should now be running on http://0.0.0.0:8080/. For stopping the docker service from the command line, please enter: bandcamplibrarian -off')
                else:
                    print('Could not find file containing environmental variables. Initializing and building Docker containers...')
                    postgres_pwd = input('Please enter a password for the Postgres database: ')
                    f = open('.vars.env', 'w')
                    f.write('POSTGRES_USER=postgres')
                    f.write('\n')
                    f.write(f'POSTGRES_PASSWORD={postgres_pwd}')
                    f.write('\n')
                    f.write('POSTGRES_DB=postgres')
                    f.write('\n')
                    flask_key=os.urandom(32).hex()
                    f.write(f'FLASK_KEY={flask_key}')                    
                    f.close()
                    print('If requested, please enter your root password for deploying the Docker services of Bandcamp Dance Librarian.')
                    try:
                        output = subprocess.check_output(['bash','-c', 'sudo docker-compose kill'])
                        output = subprocess.check_output(['bash','-c', 'sudo docker-compose rm -fv pg_container'])
                        output = subprocess.check_output(['bash','-c', 'sudo docker-compose up --detach --build --force-recreate'])
                    except:
                        output = subprocess.check_output(['bash','-c', 'docker-compose kill'])
                        output = subprocess.check_output(['bash','-c', 'docker-compose rm -fv pg_container'])
                        output = subprocess.check_output(['bash','-c', 'docker-compose up --detach --build --force-recreate'])     # for AWS deployment
                    if 'Successfully built' in output.decode("utf-8"):
                        print('The web interface should now be running on http://0.0.0.0:8080/.')
                        choice = 'x'
                        while choice not in ['y','Y','n','N']:
                            choice = input('Would you like to add the samples database including 5 labels and 2197 tracks? (y/n) ')
                        if choice in ['y','Y']:
                            import_tracks(postgres_pwd)
                        print('Please refer to the documentation/readme for further app settings (accessible via config/config.csv). For stopping the docker service from the command line, use: bandcamplibrarian -off')
                    else:
                        print('Error detected! Please refer to the Docker documentation and deploy the containers manually.')
        elif args.off:
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            print('If requested, please enter your root password for stopping the docker service of the Bandcamp Dance Librarian.')
            try:
                output = subprocess.check_output(['bash','-c', 'sudo docker-compose kill'])
            except:
                output = subprocess.check_output(['bash','-c', 'docker-compose kill'])


if __name__ == "__main__":
    main()