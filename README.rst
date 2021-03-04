========================
Bandcamp Dance Librarian
========================


.. image:: https://img.shields.io/pypi/v/bandcamp_librarian.svg
        :target: https://pypi.python.org/pypi/bandcamp_librarian

Bandcamp Dance Librarian—detecting stylistic tendencies in the Bandcamp libraries of electronic dance music labels

This project uses the subgenre taxonomy of Beatport (as of Jan 2021) in an attempt to detect stylistic tendencies or repertoires within the Bandcamp libraries of (mainly) grasroots labels. To achieve this, an automatic subgenre classifier is trained on Beatport's Top-100 lists, which identifies the possible subgenres a track may belong to based on the audio analysis of its musical features. The classifier is then applied to detect the styles a Bandcamp library may belong to. The tracks pertaining to the whole library are first individually analysed, then distributed into groups or clusters based on their possible subgenre affiliations. The project output also show the tags (folksonomies) added by the artists/labels to the Bandcamp pages. It is therefore possible to compare the industry taxonomy of Beatport with the artist folksonomies, as long as such tags are provided on Bandcamp.

A working demo of the Librarian is available under http://18.198.194.11:8080 (use http://18.198.194.11:8080/admin for the admin interface).

You can install the project from the command line by entering "pip install bandcamp-librarian".

After installation, enter "bandcamplibrarian -on" from the command line to initialize the docker service and access the web interface running on 0:0:0:0/8080. To switch off the service, use "bandcamplibrarian -off".

Alternatively, rename the .example_vars.env file to .vars.env, change POSTGRES_PASSWORD and FLASK_KEY in the file to your preferred values, and run Docker Compose from the project folder (e.g, sudo docker-compose up --detach --build).

* Free software: MIT license


Features/Pipeline
-----------------

The project pipeline is running in Docker containers featuring a Bandcamp scraper, analyser, Postgres database and a Flask-powered user/admin website.

The classifier/clusterer interface is based on a Flask application running on 0:0:0:0/8080.

The classification relies on audio features extracted individually from each label's tracks. The number of clusters can be provided in advance or determined automatically.

Results are provided in a PDF file with links to up to three representative Bandcamp tracks (if possible, from different artists) in each group. The document will also show the tags (folksonomies) added by the artists/labels to the Bandcamp pages.

New Bandcamp labels/libraries can be added by using the admin interface on 0:0:0:0/8080/admin.

The track audio features and other attributes (incl. low-res release cover data) are stored in the Postgres database.

The labels Postgres table contains a shortlist of labels including the full label name and the number of files already processed.

It is also possible to manually import the tracks and labels database by running the csvimport.py python script located in the /config folder; the csvexport.py script in the same folder can be used for exporting the database.

The config.csv file in the /config folder defines the pipeline scraping/analysis settings as well as the size limit for the scraped files (by default 20MB, approx. 20 mins long MP3 track).

To start scraping manually (i.e., not through the web interface), you can edit the config.csv: set "scraping" to 0 and "bclabel" to the bandcamp label name found in the Bandcamp url (for example, "polegroup" for "https://polegroup.bandcamp.com/"). To stop scraping manually, set "scraping" to 0 and "bclabel" to "_none_" in config.csv.

An additional parameter in config.csv: by setting "prediction_weight" to 2, the cluster/genre prediction will be aimed towards purer subgenres; by setting it to 0.5, it will be aimed towards a better amalgam of subgenres.

The classification model can be dynamically modified by replacing the beatport_classifier.sav file in the /config folder.


Methodology
-----------

The subgenre classification algorithm follows the methodology outlined in Caparrini et al. (2020), the datasets of which encompass audio features extracted from Beatport's Top-100 lists covering a range of electronic dance music subgenres.

This project relies on Beatport's Jan 2021 Top-100 lists, covering 33 categories. These included DJ Tools, which was not an actual subgenre but a collection of sound samples destined for DJs and producers - therefore, it was not included in this project.

The Electronica category, a loose collection of tracks reated to various subgenres, was replaced with 100 tracks (selected from the full range of release dates while excluding duplicate artists) out of the 439 tracks labelled as Ambient in the Beatport catalogue. Ambient is more defined in terms of intrinsic musical qualities than Electronica, while being listed as a subgenre of Electronica on Beatport. Many electronic dance music releases feature ambient tracks or influences, which warranted its inclusion into the dataset. However, ambient tracks are usually beatless, and their BPM prediction is often erroneous, while BPM values are the most important features of the model. This may confuse the prediction; to reduce the importance of this class, during model building a class weight of 0.5 was applied to Ambient.

92 audio features were extracted using pyAudioAnalysis and Essentia from the track samples provided with the lists. The resulting 3200-tracks dataset contained 17 duplicates. These were replaced with tracks/features extracted from Beatport's Dec 2020 Top-100 lists (the top track(s) from the corresponding subgenres were selected).

Location of the final dataset in the project folder structure: dataset-model-predictor/beatport_2021jan.csv

Based on this dataset, an sklearn ExtraTreesRegressor model was trained that classifies tracks based on their 92 audio features. The model model was tuned up to a 10-fold cross validation score of 0.536 (with a standard deviation of 0.023) on the training split of the dataset, reaching an F1 score of 0.531 on the testing split of the dataset. See in the project folder: dataset-model-predictor/build_model.py

Considering the number of subgenres, these results are in the range of the performance scores provided by Caparrini et al. (2020): their k=10 validation accuracies are 0.590 +/- 0.026 for the classifier trained on Beatport Set 1 (2016) containing 23 subgenres; and 0.482 +/- 0.024 for the classifier trained on Beatport Set 2 (2018) contained 29 subgenres. According to Caparrini et al. (2020) these are fair results when taking into account the standard features extracted, the high number of subgenres and subgenre proximities.

The features for the clustering algorithm are the 32 class (subgenre) probabilities provided by the classifier for each track. The K-means clustering is run multiple (20) times; during each iteration, the three highest probability values from the cluster centroids (the three subgenres that the cluster centers are most probably affiliated with) are added together with an optional weight of 0.5 or 2 applied to the highest value. Finally, the model with the highest cumulative sum of probabilities across all clusters is stored. Although this optimisation process is somewhat arbitrary, it is meant to ensure that the cluster centroids are crystallised around the classifier's subgenre categories (i.e., the confidence of prediction at the center is relatively high); the weight is applied to decrease or increase the importance of the highest value in selecting the final model, thus resulting in purer subgenre clusters or a better amalgam of subgenres. If the user selects automatic cluster number recommendation, this whole process is repeated for cluster numbers ranging from 1 to 6, and finally the cluster number located at the elbow of the inertia curve is selected. If no elbow can be defined in 5 consecutive attempts, the number of clusters is set to 4.

A PDF report is generated with the three highest subgenre probability values pertaining to the centroids and three track examples (i.e. tracks closest to their centroids based on Euclidean distance metering) in each cluster, with links to their Bandcamp pages and their associated Bandcamp folksonomies.

References

Antonio Caparrini, Javier Arroyo, Laura Pérez-Molina and Jaime Sánchez-Hernández. 2020. "Automatic subgenre classification in an electronic dance music taxonomy." Journal of New Music Research 49(12):1-16.

Leonard Kaufman and Peter J. Rousseeuw. 1990. Finding Groups in Data: An Introduction to Cluster Analysis. Hoboken, New Jersey: John Wiley & Sons.

Credits
-------

The Bandcamp scraper is based on SoundScrape / Rich Jones
Miserlou / SoundScrape - https://github.com/Miserlou/SoundScrape

This package was created with Cookiecutter_ and the
`Spiced Academy Cookiecutter PyPackage <https://github.com/spicedacademy/spiced-cookiecutter-pypackage>`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
