"""Console script for Bandcamp Dance Librarian."""

import os


def main():
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print('If requested, please enter the root password for stopping the docker service of the Bandcamp Dance Librarian.')
    os.system('sudo docker-compose kill')


if __name__ == "__main__":
    main()