#! /usr/bin/env python3
# @author	   : Pedro Mariano Sousa Bezerra <pedromsb@unicamp.br>
# @file	   	   : main.py
# @created	   : 23-Jul-2025
# @company 	   : School of Electrical and Computer Engineering - UNICAMP - Campinas - Brazil

from algo_surprise import RecAPCF
import timeit
#from numpy import mod
from data_preprocessing import create_dat, check_patterns
import subprocess

# Dataset parameters
dataset = 'ml-100k'
rating_scale = (1,5)
t_rat = 2.5 # Releavence threshold
N_metric = 25 # N for precision and recall @ N

files_dir = 'data'
files_dir = files_dir + '/' + dataset + '/'

#APCF parameters
n_splits = 5
n_pat = 20 # number of similar patterns
k_neigh = 400 # K parameter for IBKNN approach
t_bin = 0 # Binarization threshold

seed = 1
sim_name = 'adj_cosine'
save = False # Save predictions

optimize = True # Calculate similarity only for items in the pattern set for APCF
n_ass = 1 # Number of similar users for pattern selection

# BinaPs parameters
n_epochs = 50
t_size = .8 # Training set size
lr = 1e-2 # Learning rate
gamma = 0.1 # Learning step
retrain_AE = False # Set to True to retrain BinaPs each run
        
print(f'Running APCF...')

start = timeit.default_timer()

ret = check_patterns(files_dir, dataset, n_splits, t_bin) # Check if patterns are pre-trained
retrain_AE = ret | retrain_AE

if(retrain_AE):

    print('Training BinaPs...')
    for i in range(n_splits):
        filepath = files_dir + dataset + f'_fold{i}_t{int(t_bin*10)}.dat'
        try:
            with open(filepath, "r") as f:
                pass
        except FileNotFoundError:
            create_dat(files_dir, dataset, n_splits, t_bin) # Create dat files for BinaPs if necessary
            
        # Train BinaPs    
        subprocess.call(["python", "binaps/main.py", "--input", filepath, "--epochs", str(n_epochs), "--train_set_size", str(t_size), "--lr", str(lr), "--gamma", str(gamma)])
        
    stop = timeit.default_timer()
    t_ae = stop - start   
    print(f'BinaPs training completed, time elapsed: {t_ae}s')
    
else:
    t_ae = 0
    print('Using pre-trained patterns')

# CF step
algo = RecAPCF(k_neigh, n_pat, sim_name, optimize, n_ass)
algo.run_KFold(n_splits, files_dir, dataset, N_metric, t_rat=t_rat, t_bin = t_bin, sd=seed, save_pred=save, scale=rating_scale)
stop = timeit.default_timer()
t_total = stop - start
print(f'APCF completed, total runtime: {t_total}s, CF only: {t_total - t_ae}s')
