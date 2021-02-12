"""Console script for Bandcamp Dance Librarian."""

import os


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print('If requested, please enter the root password for building the docker images of the Bandcamp Dance Librarian.')
    os.system('sudo docker-compose build')
    print('If requested, please enter the root password for starting the Bandcamp Dance Librarian docker containers in the background.')
    os.system('sudo docker-compose up -d')
    print('The web interface should now be running on http://0.0.0.0:8080/. For stopping the docker service from the command line, please enter: bandcamplibrarian-off')


if __name__ == "__main__":
    main()