import numpy as np
#import h5py
from sklearn.model_selection import KFold
import pandas as pd

def add_noise(m, perc):
    
    n_cells = m.size
    n_flips = np.ceil(perc * n_cells).astype(int)
    flip_pos = np.random.choice(np.arange(n_cells), size=n_flips, replace=False)

    for i in flip_pos:
        if m.flat[i] == 0:
            m.flat[i] = 1
        else:
            m.flat[i] = 0
            
def gen_pattern(npats, patlen, nrows, density, noiselvl, files_dir = 'data'):
    
    patCols = [[] for _ in range(npats)]
    patRows = [[] for _ in range(npats)]
    nextCol = 1
    
    for pid in range(1, npats + 1):
        lnum = np.random.choice(range(2, patlen + 1))
        # assign feature ids to patterns
        patCols[pid - 1] = np.arange(nextCol, nextCol + lnum)
        nextCol += lnum

        nr = int(np.random.normal(density * nrows, density * nrows / 10))
        # redraw illegal numbers
        while nr <= 0 or nr >= nrows:
            nr = int(np.random.normal(density * nrows, density * nrows / 10))
        patRows[pid - 1] = np.random.choice(range(nrows), nr, replace=False)

    ncols = sum(len(cols) for cols in patCols)
    #print(ncols)
    m = np.zeros((nrows, ncols))
    #print(m.shape)

    # apply patterns
    for pid in range(1, npats + 1):
        m[np.ix_(patRows[pid - 1], patCols[pid - 1] - 1)] = 1
    
    # Save matrix before noise
    # filename = files_dir + '/' + dataset + '.dat'
    # matrix_to_dat(filename, m)

    # add noise
    add_noise(m, noiselvl)

    #print("Added noise.")

    # remove rows without content
    m = m[~np.all(m == 0, axis=1)]
    #print("Removed rows without content.")
    
    return {"data": m, "pats": patCols}
    
def matrix_to_dat(filename, matrix, repeat=0.25):
    
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
                
def gen_ratings(data, t_rat, ratio = 0.8, min_rating = 0.0, max_rating = 5.0, spac = 0.5):

    id_pos = np.where(data.flat == 1)[0]
    n_pos_rat = len(id_pos)
    
    n_neg_rat = int(np.random.normal(n_pos_rat*(1/ratio - 1), n_pos_rat/10))
    id_zero = np.where(data.flat == 0)[0]
    id_neg = np.random.choice(id_zero, n_neg_rat, replace=False)
    
    neg_rat = np.arange(min_rating + spac, t_rat + spac, spac)
    pos_rat = np.arange(t_rat + spac, max_rating + spac, spac)
    
    data.flat[id_neg] = np.random.choice(neg_rat, n_neg_rat)
    data.flat[id_pos] = np.random.choice(pos_rat, n_pos_rat)
     
def get_bin_data(data, t_rat):

    bin_data = np.copy(data)
    ind = (bin_data > t_rat)
    bin_data[ind] = 1
    bin_data[~ind] = 0
    bin_data = bin_data.astype(int)
    
    return bin_data
    
def save_patterns(dataset, pats, files_dir = 'data'):
    
    filename = files_dir + '/' + dataset + '.patterns'
    with open(filename, 'w') as writer:

        for p in pats[:-1]:
            for item in p[:-1]:
                writer.write(str(item) + ' ')
            writer.write(str(p[-1]) + '\n')
        
        for item in pats[-1][:-1]:
            writer.write(str(item) +' ')
        writer.write(str(pats[-1][-1]))
           
def surprise_split(dataset, matrix, X, Y, test_index, files_dir = 'data'):

    # Split data in csv files for surprise framework

    # dataset: dataset name
    # matrix: rating matrix
    # X: row indices of non-zero ratings
    # Y: columns indices of non-zero ratings
    # test_index: list of indices for the test set of each fold

    n_ratings = len(X)
    n_splits = len(test_index)
    all_index = np.arange(n_ratings)

    for i in np.arange(n_splits):
        
        train_index = np.delete(all_index, test_index[i])
        train_df = pd.DataFrame(data={'user': X[train_index],
                       'item': Y[train_index],
                       'rating' : matrix[X[train_index], Y[train_index]]})
        filename = files_dir + '/' + dataset + f'_train{i}.csv'
        train_df.to_csv(filename, sep=';', index=False)
        
        test_df = pd.DataFrame(data={'user': X[test_index[i]],
                       'item': Y[test_index[i]],
                       'rating' : matrix[X[test_index[i]], Y[test_index[i]]]})
        filename = files_dir + '/' + dataset + f'_test{i}.csv'
        test_df.to_csv(filename, sep=';', index=False)
    
def gen_data(dataset, npats, patlen, nrows, density, noiselvl, t_rat = 2.5, t_bin = 0, n_splits = 5, ratio = 0.8, min_rating = 0.0, max_rating = 5.0, spac = 0.5, 
             files_dir = 'data', save_pat = False):

    # Generate data and save

    # dataset: name of the dataset
    # npats: desired number of patterns
    # patlen: maximum number of columns in a pattern (>2)
    # nrows: maximum number of rows
    # density: average fraction of rows in a pattern (between 0.0 and 1.0)
    # noiselvl: fraction of noise (between 0.0 and 1.0)
    # t_rat: positive rating threshold
    # n_splits: number of k-folds
    # min_rating: minimum value
    # max_rating: maximum value
    # spac: spacing between two consecutive ratings

    r = gen_pattern(npats, patlen, nrows, density, noiselvl, files_dir)
    data = r['data']
    pats = r['pats']
    
    if(save_pat):
        save_patterns(dataset, pats, files_dir) # Save patterns
    
    n_rows, n_cols = data.shape # actual matrix shape
    
    gen_ratings(data, t_rat, ratio, min_rating, max_rating, spac)
    
    # Split binary data in k-folds
    kf = KFold(n_splits, shuffle=True)
    ktest = []
    
    #X, Y = index_nonzero(data)
    X, Y = np.nonzero(data)
    
    # Save dat files for each fold
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        
        ktest.append(test_index)
        X_test = X[test_index]
        Y_test = Y[test_index]
        
        kdata = get_bin_data(data, t_bin)
        kdata[X_test, Y_test] = 0.0 # Erases ratings from test set
        
        # Convert  to .dat format
        filename = files_dir + '/' + dataset + f'_fold{fold}_t{int(t_bin*10)}.dat'
        matrix_to_dat(filename, kdata)
        
    '''    
    # Save real data and test indices in h5 file
    filename = files_dir + '/' + dataset + '.h5'
    with h5py.File(filename, 'w') as f:

        f['data'] = data
        for fold in np.arange(n_splits):
            f[f'test_index{fold}'] = ktest[fold]
    '''
            
    # Generate csv files for surprise framework
    surprise_split(dataset, data, X, Y, ktest, files_dir)
                          
def main(dataset, npats, patlen, nrows, density, noiselvl, t_rat = 2.5, t_bin = 0, n_splits = 5, ratio = 0.8, min_rating = 0.0, max_rating = 5.0, spac = 0.5, seed = 1, 
         files_dir = 'data', save_pat = False):

    np.random.seed(seed)
    gen_data(dataset, npats, patlen, nrows, density, noiselvl, t_rat, t_bin, n_splits, ratio, min_rating, max_rating, spac, files_dir, save_pat)
        
if __name__ == "__main__":

    files_dir = '.' # path to save files
    dataset = 'patRec' # dataset name
    
    seed = 1
    save_pat  = False # Save original patterns before train-test split
    
    t_rat = 2.5 # Relevance threshold
    t_bin = 2.5 # Binarization threshold
    ratio = 0.5 # Positive / negative samples ratio
    n_splits = 5 # Number of folders for cross-validation
    
    # Rating parameters
    min_rating = 0.0
    max_rating = 5.0
    spac = 0.5
    
    # PatRec Parameters
    npats = 100
    patlen = 20
    nrows = 300
    density = 0.1
    noiselvl = 0.05
    
    # PatRec-large Parameters
    '''
    npats = 1000
    patlen = 20
    nrows = 30000
    density = 0.01
    noiselvl = 0.004
    '''
    
    main(dataset, npats, patlen, nrows, density, noiselvl, t_rat, t_bin, n_splits, ratio, min_rating, max_rating, spac, seed, files_dir, save_pat)
