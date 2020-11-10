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

num_stats = 13 # 13 with beta, 12 without
num_subw = 25
def reshape(data, row_num):
    data = data.iloc[row_num]
    data = data.loc['pi_win0':] # slice all stats
    data = data.values.reshape(num_stats, -1)
    return data

def save_data(worm_dir, filenames):
    # get directory and file names 
    filenames = np.loadtxt(filenames, dtype='str')
    num_sims = len(filenames)
    
    # data to be saved
    fvecs = np.zeros((num_sims, num_stats, num_subw, 3))

    for i, sim_name in enumerate(filenames):
        sim = pd.read_table(worm_dir + sim_name)
        for j in range(3):
            fvecs[i,:,:,j] = reshape(sim, j)

    with open('data/fvecs.pkl', 'wb') as fvec_pkl:
        pickle.dump(fvecs, fvec_pkl)

worm_dir = '/projects/haldane/shared/worm_landscapes/1mb_25subw_beta/SIM/'
filenames = '1mb_25subw_beta_filenames.txt'
save_data(worm_dir, filenames)
