from abc import ABC, abstractmethod
import numpy as np

# plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 200

# model building
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.io.gfile import glob, exists, mkdir, remove
from tensorflow.math import confusion_matrix
from tensorflow.random import set_seed

# data wrangling
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


##################
#   BASE MODEL   #
##################
class BaseClassifier(ABC):
    def __init__(self, model_name, balancing='keep'):
        self.model = None
        self.is_trained = False
        self.set_logpath(model_name)
        self.prep_data(balancing)

    #################
    #   MODEL FIT   #
    #################
    @abstractmethod
    def compile_model(self):
        """ construct network layers """
        raise NotImplementedError()
    
    def fit_model(self, max_epochs, verbose=False):
        """ fit to training data """
        if self.model is None:
            self.model = self.compile_model()
        else:
            self.model = self.load_best_model()

        if verbose: self.model.summary()

        # callbacks
        callbacks = self.get_callbacks()

        # train network
        y_train = [self.mut_train, self.self_train, self.sel_train]
        y_test = [self.mut_test, self.self_test, self.sel_test]
        valid_data = (self.x_test, y_test)
        set_seed(23)
        self.history = self.model.fit(self.x_train, y_train, 
                                      validation_data=valid_data, 
                                      epochs=max_epochs, 
                                      callbacks=callbacks,
                                      verbose=0)
        self.is_trained = True

        # clean checkpoints
        self.keep_top_k_models(k=5)

        # pickle training history
        with open(self.logpath + 'training_history.pkl', 'wb') as f:
            pickle.dump(self.history.history, f)
    
    ############
    #   DATA   #
    ############
    def keep_top_k_models(self, k):
        # get checkpoint filenames 
        ckpt_filepaths = glob(self.logpath + '*hdf5')

        # get checkpoint selection accuracy
        ckpts = [c.split('/')[-1] for c in ckpt_filepaths]
        selec_accs = [c.split('-')[2] for c in ckpts]
        selec_accs = [float(s.split('=')[-1]) for s in selec_accs]

        # save best model checkpoint path
        ordered_idxs = np.argsort(selec_accs)
        best_idxs = ordered_idxs[-k:]
        worst_idxs = ordered_idxs[:-k]

        # save best path, remove not top k checkpoints
        self.best_model_path = ckpt_filepaths[best_idxs[-1]]
        for i in worst_idxs:
            remove(ckpt_filepaths[i])

    def load_best_model(self):
        # highest selection accuracy model filepath
        best_path = self.best_model_path
        print(f'loading best model {best_path.split("/")[-1]}')
        best_model = load_model(best_path)
        return best_model

    def load_data(self, name):
        filename = f'data/{name}.pkl'
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def prep_data(self, balancing='keep'):
        """ 
        Load data, split to train/test, save as attributes 
            balancing should be one of 'keep', 'resamp', 'remove' 
        """
        # load in data
        self.labels = self.load_data('labels')
        self.var_keys = list(self.labels.keys())
        fvecs = self.load_data('fvecs')
        mutation = self.load_data('mutation_labels')
        selfing = self.load_data('selfing_labels')
        selection = self.load_data('selection_labels')
        
        # remove balancing sel from data
        if balancing == 'remove':
            non_bal_idxs = selection[:,-1] == 0
            fvecs = fvecs[non_bal_idxs]
            mutation = mutation[non_bal_idxs]
            selfing = selfing[non_bal_idxs]
            selection = selection[non_bal_idxs] # remove balancing rows
            selection = selection[:,:-1] # remove balancing column

        # count number of labels
        self.num_mut_labs = mutation.shape[1]
        self.num_self_labs = selfing.shape[1]
        self.num_sel_labs = selection.shape[1]

        # split to train and test sets
        split = train_test_split(fvecs, mutation, selfing, selection, 
                                 test_size=0.3, random_state=23)
        x_train, x_test, mut_train, mut_test, self_train, self_test, sel_train, sel_test = split
            
        # resample balancing for equal representation
        if balancing == 'resamp':
            # get balancing indexes
            bal_idxs, = np.where(sel_train[:,-1] == 1) # NDBa indexes
            n_not_bal = int(np.min(np.sum(sel_train[:,:-1], axis=0)))
            n_bal = len(bal_idxs)
            n_samples = n_not_bal - n_bal
            boot_idxs = resample(bal_idxs, n_samples=n_samples)

            def boot(data):
                boot_data = data[boot_idxs]
                return np.concatenate([data, boot_data], axis=0)

            # resampling
            x_train = boot(x_train)
            mut_train = boot(mut_train)
            self_train = boot(self_train)
            sel_train = boot(sel_train)
        
        self.x_train, self.x_test = x_train, x_test
        self.mut_train, self.mut_test = mut_train, mut_test
        self.self_train, self.self_test = self_train, self_test
        self.sel_train, self.sel_test = sel_train, sel_test

    def predict(self, filename):
        """ Predicts on empirical data pickled in filename """
        assert self.is_trained 

        # load empirical data
        data = self.load_data(filename)
        preds = self.model.predict(data)
        return preds

    #####################
    #   GETS AND SETS   #
    #####################
    def set_logpath(self, model_name):
        """ Sets model version number and log path """
        # make model name directory if not exist
        if not exists(f'model_logs/{model_name}'):
            mkdir(f'model_logs/{model_name}')

        # make version directory
        if not exists(f'model_logs/{model_name}/version_0'):
            vnum = 0
        else:
            # vnum is 1+ most recent vnum
            versions = glob(f'model_logs/{model_name}/version_*')
            current_vnum = max([int(v.split('_')[-1]) for v in versions])
            vnum = current_vnum + 1

        # save version number and logpath
        self.vnum = vnum
        self.logpath = f'model_logs/{model_name}/version_{vnum}/'
        mkdir(self.logpath)

    def get_callbacks(self):
        """ configures checkpoint and lr scheduler callbacks """
        # checkpoint callback
        fname = self.logpath + 'epoch={epoch:02d}'
        fname += '-mut_acc={val_mutation_accuracy:.3f}'
        fname += '-selec_acc={val_selection_accuracy:.3f}'
        fname += '-self_acc={val_selfing_accuracy:.3f}.hdf5'
        ckpt = ModelCheckpoint(filepath=fname, 
                               monitor='selection_accuracy', 
                               save_best_only=True,
                               mode='max')

        return [ckpt]

    def get_history(self, *keys):
        """ Unpacks keys from history """
        assert self.model is not None and self.history is not None
        history = self.history.history
        key_histories = []
        for key in keys:
            key_histories.append(history[key])
        return tuple(key_histories)

    def get_cm(self, truth, preds, softmax=True):
        if softmax:
            truth = np.argmax(truth, axis=1)
            preds = np.argmax(preds, axis=1)
        return confusion_matrix(truth, preds).numpy()

    def get_string_from_softmax(self, softmax_dict):
        """
        Given softmax dict with keys=(mutation, selfing, selection)
        returns the string representation of each prediction
        """
        # convert softmax predictions to argmax then strings
        str_dict = softmax_dict.copy()
        for var, str_labs in self.labels.items():
            argmax = np.argmax(softmax_dict[var], axis=1)
            str_dict[var] = np.array([str_labs[i] for i in argmax])
        return str_dict

    def get_prediction_dict(self):
        """ 
        Predicts on test data
        Returns dictionary with var strings as keys
        and string label predictions as values
        """
        assert self.is_trained
        model = self.load_best_model()
        # predict with model, convert to dict, get string labels
        softmax_preds = model.predict(self.x_test)
        softmax_preds_dict = dict(zip(self.var_keys, softmax_preds))
        preds = self.get_string_from_softmax(softmax_preds_dict)

        # get ground truth string labels
        softmax_truth = [self.mut_test, self.self_test, self.sel_test]
        softmax_truth_dict = dict(zip(self.var_keys, softmax_truth))
        truth = self.get_string_from_softmax(softmax_truth_dict)

        # merge to single dictionary
        prediction_results = dict()
        for var in self.var_keys:
            var_truth = f'{var}_truth'
            var_pred = f'{var}_pred'
            prediction_results[var_truth] = truth[var]
            prediction_results[var_pred] = preds[var]
        return prediction_results
        
    def get_misclass_dicts(self, cond_on='mutation'):
        """ 
        Given a variable cond_on in [mutation, selfing, selection] 
        to condition on being misclassified, 
        get string predictions for not cond_on variables
        """
        # check cond_on is correctly chosen
        assert cond_on in self.var_keys

        # get predictions as strings
        str_preds = self.get_prediction_strings()

        # store test labels as strings
        test = [self.mut_test, self.self_test, self.sel_test]
        test_dict = dict(zip(self.var_keys, test))
        str_test_dict = self.get_string_from_softmax(test_dict)
        
        # get misclassified indexes
        truth = str_test_dict[cond_on]
        preds = str_preds[cond_on]
        misclass_idx, = np.where(truth != preds)
        
        # filter misclassified cond_on predictions
        misclass_truth = str_test_dict.copy()
        misclass_preds = str_preds.copy()
        for var in self.var_keys:
            misclass_truth[var] = misclass_truth[var][misclass_idx]
            misclass_preds[var] = misclass_preds[var][misclass_idx]

        return misclass_truth, misclass_preds

    def get_misclass_array(self, cond_on='mutation'):
        """
        Given a cond_on variable in [mutation, selfing, selection]
        returns square array of size len(cond_on) where the rows
        are true cond_on classes, columns are predicted cond_on classes
        and entries are indexes to slice test with for (true, pred) pair
        """
        # check cond_on is correctly chosen
        assert cond_on in self.var_keys
        not_cond_on = [x for x in self.var_keys if x != cond_on]
        misclass_dict = {x: None for x in not_cond_on}
        truth, preds = self.get_misclass_dicts(cond_on)

        for var in not_cond_on:
            pass
        
    ################
    #   PLOTTING   #
    ################
    
    # TODO 
    # add final version of plot_var_counts
    # confirm best model is loaded for model fit plots

    def plot_misclass_cm(self, cond_on='mutation'):
        """
        Predicts on test set and consider cond_on misclassifications
        and plots confustion matrices of not cond_on variables
        """
        data = self.get_misclass(cond_on)
        cond_on_labs, vardicts = zip(*data.items())
        varnames = vardicts[0].keys()

        # plot misclass
        nrows = len(data)
        ncols = 2
        fig, ax = plt.subplots(nrows, ncols, dpi=200)

        for i, cond_on_lab in enumerate(cond_on_labs): # will be 1, 1.15, 2
            for j, other_var in enumerate(varnames): # will be 
                ax[i,j].hist(data[cond_on_lab][other_var])
                ax[i,j].set_xlabel(other_var)
                ax[i,j].set_ylabel(cond_on_lab)
        filename = f'{cond_on}_misclass_{not_cond_on}_cm.png'
        plt.savefig(self.logpath + filename)
        plt.close()

    def plot_cm(self, cm, classes, filename, 
                normalize=False, 
                cmap=plt.cm.Blues):
        import itertools
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f'{filename} confusion matrix')
        print(cm_norm)
        if normalize: cm = cm_norm

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('%s confusion matrix' % filename, fontsize=15)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        rows, cols = cm.shape
        for i, j in itertools.product(range(rows), range(cols)):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.logpath + f'{filename}.png')
        plt.close()

    def plot_confusion_matrix(self):
        assert self.is_trained
        # load best checkpoint
        self.model = self.load_best_model()

        preds = self.model.predict(self.x_test)
        labels = self.load_data('labels')

        # plot mutation confusion matrix
        mut_cm = self.get_cm(self.mut_test, preds[0])
        mut_labs = labels['mutation']
        self.plot_cm(mut_cm, mut_labs, 'mutation')

        # plot selfing confusion matrix
        self_cm = self.get_cm(self.self_test, preds[1])
        self_labs = labels['selfing']
        self.plot_cm(self_cm, self_labs, 'selfing')

        # plot selection confusion matrix
        sel_cm = self.get_cm(self.sel_test, preds[2])
        sel_labs = labels['selection']
        self.plot_cm(sel_cm, sel_labs, 'selection')

    def plot_history(self, key):
        """ plot training and validation history unless learning rate """
        assert key in self.history.history.keys()
        fig, ax = plt.subplots(dpi=200)
        if key == 'lr':
            lr, = self.get_history(key)
            ax.plot(lr, label='learning rate')
            plt.xlabel('epoch')
            plt.ylabel('learning rate')
        else:
            train_key, valid_key = key, f'val_{key}'
            training, validation = self.get_history(train_key, valid_key)
            ax.plot(training, label='training')
            ax.plot(validation, label='validation')
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.legend()

        # save in logpath
        plt.savefig(self.logpath + f'{key}.png')
        plt.close()

    def plot_loss(self):
        """ Plot and save training losses """
        for k in ['loss', 'mutation_loss', 'selfing_loss', 'selection_loss']:
            self.plot_history(k)

    def plot_acc(self):
        """ Plot and save training accuracies """
        for k in ['mutation_accuracy', 'selfing_accuracy', 'selection_accuracy']:
            self.plot_history(k)

    def plot_lr(self):
        """ Plot and save learning rate vs epochs """
        self.plot_history('lr')


class VanillaClassifier(BaseClassifier):
    def __init__(self, balancing='keep'):
        super().__init__('vanilla', balancing)

    def compile_model(self):
        # build network
        x = Input(self.x_train[0,:].shape)

        y = Conv2D(6, (2, 5), activation='relu')(x)
        y = Conv2D(12, (3, 5), activation='relu')(y)
        y = Conv2D(24, (4, 5), activation='relu')(y)
        y = Conv2D(48, (5, 5), activation='relu')(y)
        y = Flatten()(y)
        y = Dense(100, 'relu')(y)
        y = Dense(20, 'relu')(y)

        # concatenate last hidden layers
        mut_pred = Dense(self.num_mut_labs, 'softmax', name='mutation')(y)
        self_pred = Dense(self.num_self_labs, 'softmax', name='selfing')(y)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(y)

        model = Model(x, [mut_pred, self_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        model.compile('adam', losses, loss_weights=[1.0, 1.0, 1.0],
                      metrics=['accuracy'])
        return model

##############
#   MODELS   #
##############
class ParallelClassifier(BaseClassifier):
    def __init__(self, balancing='keep'):
        super().__init__('parallel', balancing)

    def compile_model(self):
        # build network
        x = Input(self.x_train[0,:].shape)

        def build_branch(x):
            y = Conv2D(6, (2, 5), activation='relu')(x)
            y = Dropout(0.2)(y)
            y = Conv2D(12, (3, 5), activation='relu')(y)
            y = Conv2D(24, (4, 5), activation='relu')(y)
            y = Dropout(0.2)(y)
            y = Conv2D(48, (5, 5), activation='relu')(y)
            y = Flatten()(y)
            y = Dense(100, 'relu')(y)
            y = Dropout(0.1)(y)
            y = Dense(20, 'relu')(y)
            return y

        # network branches
        m = build_branch(x) # mutation branch
        s = build_branch(x) # selfing branch
        sel = build_branch(x) # selection branch

        # concatenate last hidden layers
        mss = concatenate([m, s, sel])
        mut_pred = Dense(self.num_mut_labs, 'softmax', name='mutation')(mss)
        self_pred = Dense(self.num_self_labs, 'softmax', name='selfing')(mss)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(mss)

        model = Model(x, [mut_pred, self_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        model.compile('adam', losses, loss_weights=[1.0, 1.0, 1.0],
                      metrics=['accuracy'])
        return model


class BounceClassifier(BaseClassifier):
    def __init__(self, balancing='keep'):
        super().__init__('bounce', balancing)

    def compile_model(self):
        # input template
        x = Input(self.x_train[0,:].shape)

        def conv_branch(x, channels, kernel_size):
            y = Conv2D(channels, kernel_size, activation='relu')(x)
            return y

        def dense_branch(x, units):
            y = Dense(units, activation='relu')(x)
            return y

        def conv_bounce(x, channels, kernel_size):
            y1 = conv_branch(x, channels, kernel_size)
            y2 = conv_branch(x, channels, kernel_size)
            y3 = conv_branch(x, channels, kernel_size)
            y = concatenate([y1, y2, y3])
            return y

        def dense_bounce(x, units, flatten=False):
            if flatten:
                x = Flatten()(x)
            y1 = dense_branch(x, units)
            y2 = dense_branch(x, units)
            y3 = dense_branch(x, units)
            y = concatenate([y1, y2, y3])
            return y

        # build network
        y = conv_bounce(x, 6, (2, 5))
        y = Dropout(0.2)(y)
        y = conv_bounce(y, 12, (3, 5))
        y = Dropout(0.2)(y)
        y = conv_bounce(y, 24, (4, 5))
        y = Dropout(0.2)(y)
        y = conv_bounce(y, 48, (5, 5))
        y = dense_bounce(y, 500, flatten=True)
        y = Dropout(0.1)(y)
        y = dense_bounce(y, 100)
        y = Dropout(0.1)(y)
        y = dense_bounce(y, 20)

        # final layer predictions
        mut_pred = Dense(self.num_mut_labs, 'softmax', name='mutation')(y)
        self_pred = Dense(self.num_self_labs, 'softmax', name='selfing')(y)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(y)

        model = Model(x, [mut_pred, self_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        model.compile('adam', losses, loss_weights=[1.0, 1.0, 1.0],
                      metrics=['accuracy'])
        return model
