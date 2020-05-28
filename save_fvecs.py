import numpy as np 
import pandas as pd
import os

# load fvec
# has 16 columns, last 12 are summary stats
# first 4 are chrom, classifiedWinStart, classifiedWinEnd, bigWinRange
# number of rows are number of windows

# so each window has a single feature vector associated to it
# we want to distinguish between uniform mut rate or 2x mut rate on arms
# using the signal from scanning across the chromosome
# thus it makes sense to use (num_wins, num_stats, num_regions) as inputs
# where num_regions is 3 for left arm, centromere, right arm

# each file will have different total rows 
# so we first subset them all to have the intersection of 
# all file rows, with respect to classifiedWinStart

# get directory and file names 
worm_dir = '/projects/haldane/shared/worm_landscapes/10kb/'
file_names = np.loadtxt('2kb_file_names.txt', dtype='str')

def load_wins(file_name):
    wins = np.loadtxt(worm_dir + file_name,
                      delimiter='\t',
                      usecols=(3,),
                      dtype='str')
    return wins[1:]

# get intersection indices for first two files
# and intersect it with next file
def intersect_wins(file_names):
    """
    Compares bigWinRange between adjacent fvec files
    and returns their intersection
    """
    # load the first two files
    old_wins = load_wins(file_names[0])
    new_wins = load_wins(file_names[1])
    
    # intersect the files
    old_and_new = np.intersect1d(old_wins, new_wins)

    # get cumulative intersection
    for file_name in file_names[2:]:
        new_wins = load_wins(file_name)
        old_and_new = np.intersect1d(old_and_new, new_wins)

    return old_and_new

common_rows = intersect_wins(file_names)
print(common_rows)
