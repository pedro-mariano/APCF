import numpy as np
import pandas as pd

def get_data(df, n_rows, n_cols):
# Returns rating matrix from dataframe
    
    data = np.zeros((n_rows, n_cols))
    for row in df.itertuples(index=False):
        data[row[0], row[1]] = row[2]
    
    return data
    
def matrix_to_dat(filename, matrix, repeat=0.25):
# Creates dat file from rating matrix
# repeat: fraction of lines to be repeated in the end of file
    
    with open(filename, 'w') as writer:
        writer.write('')
        
    n_rows, n_cols = matrix.shape

    for i in np.arange(n_rows):
        
        with open(filename, 'a') as writer:
            
            ind = np.arange(n_cols)[matrix[i] > 0]
            if(ind.size > 0):
                for item in ind[:-1]:
                    writer.write(str(item + 1)+' ')
                writer.write(str(ind[-1] + 1)+'\n')
                
    n_repeat = int(repeat*n_rows)
    
    with open(filename, 'r') as reader:
        r_lines = reader.readlines()[:n_repeat]
    
    with open(filename, 'a') as writer:
        for line in r_lines:
            writer.write(line)
            
def get_shape(filepath, dataset):
# Get rating matrix shape

    train_path = filepath + dataset + f'_train0.csv'
    df_train = pd.read_csv(train_path, sep=';')
    test_path = filepath + dataset + f'_test0.csv'
    df_test = pd.read_csv(test_path, sep=';')
    df_complete = pd.concat([df_train, df_test])
    n_rows = max(df_complete['user']) + 1
    n_cols = max(df_complete['item']) + 1
    
    return n_rows, n_cols
    
def create_dat(filepath, dataset, n_splits, t_bin):
# Creates dat files from csv

    n_rows, n_cols = get_shape(filepath, dataset)

    for i in np.arange(n_splits):
        path = filepath + dataset + f'_train{i}.csv'
        df = pd.read_csv(path, sep=';')
        data = get_data(df, n_rows, n_cols)
        bin_ratings = np.copy(data)
        ind = (data > t_bin)
        bin_ratings[ind] = 1
        bin_ratings[~ind] = 0
        bin_ratings = bin_ratings.astype(int)
        path = filepath + dataset + f'_fold{i}_t{int(10*t_bin)}.dat'
        matrix_to_dat(path, bin_ratings)
        
def check_patterns(filepath, dataset, n_splits, t_bin):
    
    retrain_AE = False
    
    # Check if patterns are pre-trained
    for i in range(n_splits):
        path = filepath + dataset + f'_fold{i}_t{int(t_bin*10)}.binaps.patterns'
        try:
            with open(path, "r") as f:
                pass
        except FileNotFoundError:
            retrain_AE = True
            break
            
    return retrain_AE
    
if __name__ == "__main__":

    files_dir = 'data'
    dataset = 'ml-100k'
    n_splits = 5
    t_bin = 0
    
    filepath = files_dir + '/' + dataset + '/'
    
    create_dat(filepath, dataset, n_splits, t_bin)
