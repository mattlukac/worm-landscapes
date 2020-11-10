import numpy as np 
import pandas as pd
import pickle

# load fvec
# first 4 cols are 
# chrom, classifiedWinStart, classifiedWinEnd, bigWinRange

# we want to distinguish between uniform mut rate or 2x mut rate on arms
# using the signal from scanning across the chromosome
# thus it makes sense to use (num_stats, num_subw, num_regions) as inputs
# where num_regions is 3 for left arm, centromere, right arm

n_stats = 13 # 13 with beta, 12 without
n_subw = 25
def reshape(data, row_num):
    data = data.iloc[row_num]
    data = data.loc['pi_win_0':] # slice all stats
    data = data.values.reshape(n_stats, -1)
    return data

def save_data(worm_dir, filenames, chrom):
    # get directory and file names 
    filenames = np.loadtxt(filenames, dtype='str')
    # filter out desired chromosome
    fnames = []
    for fname in filenames:
        ch = fname.split('_')[4]
        if ch == chrom:
            fnames.append(fname)

    n_sims = len(fnames)
    
    # data to be saved
    fvecs = np.zeros((n_sims, n_stats, n_subw, 3))

    for i, fname in enumerate(fnames):
        sim = pd.read_table(worm_dir + fname)
        for j in range(3):
            fvecs[i,:,:,j] = reshape(sim, j)

    with open(f'data/CE_chr_{chrom}_fvecs.pkl', 'wb') as fvec_pkl:
        pickle.dump(fvecs, fvec_pkl)

worm_dir = '/projects/haldane/shared/worm_landscapes/1mb_25subw_beta/CE/'
filenames = '1mb_25subw_beta_CE_filenames.txt'
for ch in ['I', 'II', 'III', 'IV']:
    save_data(worm_dir, filenames, chrom=ch)
