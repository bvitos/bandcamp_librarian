#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['docker',
        'docker-compose',
        'numpy',
        'pandas',
        'sqlalchemy',
        'psycopg2-binary',
        'scikit-learn',]

setup(
    author="Botond Vitos",
    author_email='boti@vitos.tv',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="This project detects stylistic tendencies in the Bandcamp libraries of electronic dance music labels by defining clusters based on Beatport's subgenre categories as of Jan 2021. To deploy the web interface on 0.0.0.0:8080/, enter bandcamplibrarian -on from the command line after installation.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bandcamp_librarian',
    name='bandcamp_librarian',
    packages=find_packages(include=['bandcamp_librarian','bandcamp_librarian.*']),
    url='https://github.com/bvitos/bandcamp_librarian',
    version='0.1.4',
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'bandcamplibrarian=bandcamp_librarian.cli:main',
        ],
    }
)
