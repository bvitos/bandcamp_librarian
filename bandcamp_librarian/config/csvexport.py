#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exports the tracks table into a .cvs file
"""

from sqlalchemy import create_engine
import pandas as pd
import psycopg2

print("Please note that you need a running instance of the Postgres docker container for the CSV export")
pguser = input("User name:")
pgpassword = input("Password:")
pg = create_engine(f'postgres://{pguser}:{pgpassword}@0.0.0.0:5555/postgres') # pg connect
tracks = pd.read_sql_table('tracks', pg)
tracks.to_csv('tracks.csv', index=False)
print("Table exported to tracks.csv")
pg.dispose()