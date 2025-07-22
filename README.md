The Autoencoder Pattern-based Collaborative Filtering (APCF) method is a Recommender System that calculates predictions using subsets of relevant items found by an efficient autoencoder technique, the BinaPs algorithm (Fischer & Vreeken, 2001).

Instructions

To install the required librarires, run the following command on a terminal:

    $ pip install -r requirements.txt

Then, you need to build the similarity module with cyhton:

    $ cython APCF_sims.pyx

    $ python setup.py build_ext --inplace

Datasets are available at: https://doi.org/10.25824/redu/40OKYE

Data files must be CSV files named in the format prefix_trainX.csv and prefix_testX.csv for training and test set data, respectively, where prefix is the dataset name, X is a number from 0 to N-1, and N is the number of folders for cross-validation. Each line of the csv files must be in the format 'user_id; item_id; rating'. The files must be placed in a directory with the dataset name, which must be placed in the 'data' directory. Then, change the appropriate dataset and APCF parameters in the main file before running it.

If you use this method, please cite the following reference:

BinaPs paper: https://doi.org/10.1145/3447548.3467348
