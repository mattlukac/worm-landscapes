import numpy as np 
import re
import pickle
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

""" pickles labels from file names """

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

if __name__ == '__main__':
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
