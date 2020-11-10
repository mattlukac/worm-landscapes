""" Wrangles feature vectors and one-hot labels """
import numpy as np
import pickle
import pandas as pd
import argparse
import re
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

# TODO
# dry this code

# get cli arguments
# user should specify the data to be wrangled
# one of: sim_fvecs, sim_labels, ce_fvecs, cr_fvecs
parser = argparse.ArgumentParser(description='Identify data to be wrangled')
parser.add_argument('--data', help='specify the type of data to be wrangled')
args = parser.parse_args()
args.data = args.data.lower()
assert args.data in ['sim_fvecs', 'sim_labels', 'ce_fvecs', 'cr_fvecs']

if args.data == 'sim_fvecs':
    wrangle_sim_fvecs()
elif args.data == 'sim_labels':
    wrangle_sim_labels()
elif args.data == 'ce_fvecs':
    wrangle_ce_fvecs()
elif args.data == 'cr_fvecs':
    wrangle_cr_fvecs()
else:
    print('wrong data specifier')

def wrangle_sim_fvecs():
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

def wrangle_sim_labels():
    """ pickles labels from file names """
    filenames = np.loadtxt('1mb_25subw_beta_filenames.txt', dtype='str')
    # test selection labelling
    def test_selection():
        selection = []
        for fname in filenames:
            fname = filter_name(fname)
            selection.append(selection_label(fname))
        print(Counter(selection))

    # test_selection()
    save_classes()

# filename parameters
param_names = ['Self', 
               'Mut', 
               'SDA', 
               'SDC', 
               'SBA', 
               'SBC', 
               'SBaA', 
               'SBaC']
num_param = len(param_names)
stype = {'Neutral': 0,
         'Neutral & Deleterious': 1,
         'Neutral & Deleterious & Beneficial': 2,
         'Neutral & Deleterious & Balancing': 3}

def filter_name(fname):
    """ 
    removes the sim_idx_Ne_5000 from beginning
    and .1mb_* from the end of a given filename
    """
    fname_list = fname.split('_')
    # extract label names and values
    for i, word in enumerate(fname_list):
        # start of filtered name
        if word == 'Self':
            start = i
        # end of filtered name
        if re.search('.+\.1mb', word):
            last_word = word.split('.')
            assert len(last_word) == 2 # check for decimal in value
            end = i + 1
            fname_list[i] = last_word[0]
    fname_list = fname_list[start:end]
    return fname_list

def selfing_class(fname):
    return fname[1]

def mutation_class(fname):
    return fname[3]

def selection_class(fname):
    """ classifies selection according to classes.txt """
    sel_names = param_names[2:]
    sel_bool = [False]*len(sel_names)
    sel = dict(zip(sel_names, sel_bool))
    # fill selection dictionary
    for i, word in enumerate(fname):
        if word in sel_names and fname[i+1] != '0':
            sel[word] = True

    # now determine selection class
    deleterious = [sel['SDA'], sel['SDC']]
    beneficial = [sel['SBA'], sel['SBC']]
    balancing = [sel['SBaA'], sel['SBaC']]
    selection = None

    # neutrality
    if not any(sel.values()):
        selection = 'Neutral'
    # neutral + deleterious
    elif all(deleterious) and not any(beneficial + balancing):
        selection = 'Neutral & Deleterious'
    # neutral + deleterious + beneficial
    elif all(deleterious + beneficial) and not any(balancing):
        selection = 'Neutral & Deleterious & Beneficial'
    # neutral + deleterious + balancing
    elif all(deleterious + balancing) and not any(beneficial):
        selection = 'Neutral & Deleterious & Balancing'
    assert selection is not None
    return selection


def save_classes():
    # get directory and file names 
    filenames = np.loadtxt('1mb_25subw_beta_filenames.txt', dtype='str')
    num_sims = len(filenames)
    
    # each parameter has a column filled with classes from each file
    param_names = ['selfing', 'mutation', 'selection']
    num_params = len(param_names)
    file_classes = [np.zeros((num_sims, 1)) for _ in range(num_params)]
    param_classes = dict(zip(param_names, file_classes))
    
    # get parameter class from filename
    for i, fname in enumerate(filenames):
        fname = filter_name(fname)
        param_classes['selfing'][i] = selfing_class(fname)
        param_classes['mutation'][i] = mutation_class(fname)
        param_classes['selection'][i] = stype[selection_class(fname)]

    lab_cats = dict()
    for param, classes in param_classes.items():
        print(f'{param} categories')
        ohe = OneHotEncoder(sparse=False)
        ohe_fit = ohe.fit(classes)
        print(ohe_fit.categories_)
        lab_cats[param] = ohe_fit.categories_
        ohe_labs = ohe_fit.transform(classes)
        print(ohe_labs.shape)
        with open(f'data/{param}_labels.pkl', 'wb') as f:
            pickle.dump(ohe_labs, f)
    with open('data/label_categories.pkl', 'wb') as f:
        pickle.dump(lab_cats, f)


def wrangle_ce_fvecs():
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

def wrangle_cr_fvecs():
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

        with open(f'data/CR_chr_{chrom}_fvecs.pkl', 'wb') as fvec_pkl:
            pickle.dump(fvecs, fvec_pkl)

    worm_dir = '/projects/haldane/shared/worm_landscapes/1mb_25subw_beta/CR/'
    filenames = '1mb_25subw_beta_CR_filenames.txt'
    for ch in ['I', 'II', 'III', 'V']:
        save_data(worm_dir, filenames, chrom=ch)
