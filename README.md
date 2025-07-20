The Autoencoder Pattern-based Collaborative Filtering (APCF) method is a Recommender System that calculates predictions using subsets of relevant items found by an efficient autoencoder technique, the BinaPs algorithm (Fischer & Vreeken, 2001).

Instructions

To install the required librarires, run:

$ pip install -r requirements.txt

Then, you need to build the similarity module with cyhton:

$ cython APCF_sims.pyx

$ python setup.py build_ext --inplace

Datasets are available at: https://doi.org/10.25824/redu/40OKYE

BinaPs paper: https://doi.org/10.1145/3447548.3467348
