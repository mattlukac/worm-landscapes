from abc import ABC, abstractmethod
import numpy as np

# plotting
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['figure.dpi'] = 200

# model building
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adadelta, Adam, Adagrad, Nadam
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.io.gfile import glob, exists, mkdir, remove
from tensorflow.math import confusion_matrix
from tensorflow.random import set_seed

# data wrangling
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# TODO
# transfer learning classifier

class PrintProgress(Callback):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.epoch_step = int(max_epochs * 0.1)
        self.percent = 0.
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        # print epoch and logs every 5 or 10 percent
        if epoch % self.epoch_step == 0:
            print(f'epoch {epoch}/{self.max_epochs}')


##################
#   BASE MODEL   #
##################
class BaseClassifier(ABC):
    def __init__(self, 
                 model_name,         # for model_log dir 
                 balancing='remove', # keep, remove, resample 
                 mutation='keep',    # keep, remove 2x mutation rate
                 unstack=False):     # stack or unstack fvecs 
        self.model = None
        self.is_trained = False
        self.set_logpath(model_name)
        self.prep_data(balancing, mutation, unstack)

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
        callbacks = self.get_callbacks(max_epochs)

        # train network
        y_train = [self.mut_train, self.slf_train, self.sel_train]
        y_test = [self.mut_test, self.slf_test, self.sel_test]
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
    def load_fvecs_and_targets(self):
        # load in data
        self.labels = self.load_data('labels')
        self.var_keys = list(self.labels.keys())
        fvecs = self.load_data('fvecs')
        mutation = self.load_data('mutation_labels')
        selfing = self.load_data('selfing_labels')
        selection = self.load_data('selection_labels')

        return fvecs, mutation, selfing, selection
        
    def unstack_fvecs(self, fvecs):
        n1, n2, n3, n4 = fvecs.shape
        flat_fvecs = np.zeros((n1, n2, n3*n4))
        for i, fvec in enumerate(fvecs):
            flat_fvecs[i] = np.reshape(fvec, (n2, n3*n4), 'F')
        fvecs = np.expand_dims(flat_fvecs, 3)
        return fvecs

    def remove_balancing(self, data):
        """ Remove balancing selection from dataset """
        fvecs, mutation, selfing, selection = data

        non_bal_idxs = selection[:,-1] == 0
        fvecs = fvecs[non_bal_idxs]
        mutation = mutation[non_bal_idxs]
        selfing = selfing[non_bal_idxs]
        selection = selection[non_bal_idxs] # remove balancing rows
        selection = selection[:,:-1] # remove balancing column
        self.labels['selection'] = self.labels['selection'][:-1]

        return fvecs, mutation, selfing, selection

    def remove_2x_mutation(self, data):
        """ Remove 2-1-2 mutation rate from dataset """
        fvecs, mutation, selfing, selection = data

        mut_idxs = mutation[:,-1]==0
        fvecs = fvecs[mut_idxs]
        mutation = mutation[mut_idxs] # remove 2-1-2 rows
        mutation = mutation[:,:-1] # remove 2-1-2 column
        selfing = selfing[mut_idxs]
        selection = selection[mut_idxs]
        self.labels['mutation'] = self.labels['mutation'][:-1]

        return fvecs, mutation, selfing, selection
        
    def resample_balancing(self, split_data):
        (x_train, x_test,
         mut_train, mut_test,
         slf_train, slf_test,
         sel_train, sel_test) = split_data

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
        slf_train = boot(slf_train)
        sel_train = boot(sel_train)

        # pack data
        split_data = (x_train, x_test,
                      mut_train, mut_test,
                      slf_train, slf_test,
                      sel_train, sel_test
                     )
        return split_data
        
    def prep_data(self, balancing='keep', mutation='keep', unstack=False):
        """ 
        Load data, split to train/test, save as attributes 
            balancing should be one of 'keep', 'resamp', 'remove' 
            mutation should be one of 'keep', 'remove'
        """
        # load in data
        data = self.load_fvecs_and_targets()
        
        # remove balancing selection from data
        if balancing == 'remove':
            data = self.remove_balancing(data)
        # remove 2-1-2 mutation from data
        if mut == 'remove':
            data = self.remove_2x_mutation(data)

        # unpack data
        fvecs, mutation, selfing, selection = data
        if unstack:
            fvecs = self.unstack_fvecs(fvecs)

        # count number of labels
        self.num_mut_labs = mutation.shape[1]
        self.num_slf_labs = selfing.shape[1]
        self.num_sel_labs = selection.shape[1]

        # split to train and test sets
        split_data = train_test_split(fvecs, mutation, selfing, selection, 
                                 test_size=0.3, random_state=23)
        # resample balancing for equal representation
        if balancing == 'resamp':
            split_data = self.resample_balancing(split_data)
        
        # unpack split data
        (self.x_train, self.x_test,
         self.mut_train, self.mut_test,
         self.slf_train, self.slf_test,
         self.sel_train, self.sel_test) = split_data
            
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

    def get_callbacks(self, max_epochs):
        """ configures checkpoint and lr scheduler callbacks """
        # checkpoint callback
        fname = self.logpath + 'epoch={epoch:02d}'
        fname += '-mut_acc={val_mutation_accuracy:.3f}'
        fname += '-selec_acc={val_selection_accuracy:.3f}'
        fname += '-self_acc={val_selfing_accuracy:.3f}.hdf5'
        ckpt = ModelCheckpoint(filepath=fname, 
                               monitor='mutation_accuracy', 
                               save_best_only=True,
                               mode='max')
        prog = PrintProgress(max_epochs)

        return [ckpt, prog]

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
        softmax_truth = [self.mut_test, self.slf_test, self.sel_test]
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

    ################
    #   PLOTTING   #
    ################
    
    def plot_var_counts(self, cond_on='selection', plot_truth=False):
        """
        Visualization of 4D lattice where 
        for each (cond_on_truth, cond_on_pred) pair of classes
        there is a cross section matrix with var1_classes rows
        and var2_classes cols.
        The color represents relative frequencies within each cross section
        The numbers represent frequency of that (var1, var2) pair
        among cond_on_truth
        """
        # get condition on truth, predictions, misclassified boolean, labels
        pred_results = self.get_prediction_dict()
        cond_on_truth = pred_results[f'{cond_on}_truth'] # obs labels
        cond_on_pred = pred_results[f'{cond_on}_pred'] # obs pred labels
        cond_on_misclass = cond_on_truth!=cond_on_pred # boolean
        cond_on_labs = self.labels[cond_on] # unique labels
        n_labs = len(cond_on_labs) # number of unique labels
        
        # get not cond on variable names and unique labels
        im_vars = [x for x in self.var_keys if x!=cond_on]
        im_labs = [self.labels[var] for var in im_vars]
            
        # plot data
        image_rows, image_cols = [len(im_lab) for im_lab in im_labs]
        tensor_shape = (n_labs, n_labs, image_rows, image_cols)
        image_tensor_counts = np.zeros(tensor_shape)
        image_tensor_freqs = np.zeros_like(image_tensor_counts)
        
        # plot 2d hists
        fig, ax = plt.subplots(n_labs, n_labs, 
                                dpi=200, 
                                sharex=True, sharey=True)
        for i in range(n_labs): # truth index
            for j in range(n_labs): # prediction index
                # row and column labels and counts 
                true_class = cond_on_labs[i]
                pred_class = cond_on_labs[j]
                if i==n_labs-1: ax[i,j].set_xlabel(pred_class, fontsize=12)
                if j==0: ax[i,j].set_ylabel(true_class, fontsize=12)
                row_bool = cond_on_truth==true_class
                col_bool = cond_on_pred==pred_class
                plot_bool = np.logical_and(row_bool, col_bool)
                
                # set ticks
                ax[i,j].set_xticks(range(len(im_labs[1])))
                ax[i,j].set_xticklabels(im_labs[1])
                ax[i,j].set_yticks(range(len(im_labs[0])))
                ax[i,j].set_yticklabels(im_labs[0])
                
                # make image of count data
                for ii in range(image_rows): # var1 index
                    for jj in range(image_cols): # var2 index
                        if plot_truth:
                            var1 = pred_results[f'{im_vars[0]}_truth']
                            var2 = pred_results[f'{im_vars[1]}_truth']
                            fig_title = f'{im_vars[0]} and {im_vars[1]} joint ground truth'
                        else:
                            var1 = pred_results[f'{im_vars[0]}_pred']
                            var2 = pred_results[f'{im_vars[1]}_pred']
                            fig_title = f'{im_vars[0]} and {im_vars[1]} joint predictions'
                        var1_bool = var1==im_labs[0][ii]
                        var2_bool = var2==im_labs[1][jj]
                        var1_and_var2 = np.logical_and(var1_bool, var2_bool)
                        marginal_class = np.logical_and(plot_bool, var1_and_var2)
                        class_count = int(sum(marginal_class))
                        image_tensor_counts[i,j,ii,jj] = class_count
                
        # make frequency data
        sums = np.sum(image_tensor_counts, axis=1, keepdims=True) # keepdims for broadcasting
        image_tensor_freqs = np.divide(image_tensor_counts, sums, 
                                       out=image_tensor_freqs, 
                                       where=sums!=0)
        for i in range(n_labs):
            for j in range(n_labs):
                for ii in range(image_rows):
                    for jj in range(image_cols):
                        number = round(image_tensor_freqs[i,j,ii,jj], 2)
                        thresh = image_tensor_counts[i,j,:,:].max() / 2
                        color = 'white' if int(thresh)==0 else 'white'
                        color = 'white' if image_tensor_counts[i,j,ii,jj]<=thresh else 'black'
                        ax[i,j].text(jj, ii, number, 
                                     ha='center', 
                                     va='center', 
                                     color=color)
                ax[i,j].imshow(image_tensor_counts[i,j,:,:])
        fig.tight_layout()
        
        # padding dicts
        title = {'mutation': 1., 'selfing': 0.9, 'selection': 0.82}
        xtitle = {'mutation': -0.02, 'selfing': 0.09, 'selection': 0.13}
        ytitle = {'mutation': 0.02, 'selfing': -0.02, 'selection': -0.03}
        hspace = {'mutation': 0.05, 'selfing': -0.6, 'selection': -0.7}
        wspace = {'mutation': -0.4, 'selfing': 0.1, 'selection': 0.1}
        
        # axis titles
        fig.text(0.5, title[cond_on], fig_title, 
                    ha='center', 
                    fontsize=20)
        fig.text(0.5, xtitle[cond_on], f'predicted {cond_on}', 
                    ha='center', 
                    fontsize=15)
        fig.text(ytitle[cond_on], 0.5, f'true {cond_on}', 
                    va='center', 
                    rotation='vertical', 
                    fontsize=15)
        
        plt.subplots_adjust(hspace=hspace[cond_on], wspace=wspace[cond_on])
        plt.savefig(self.logpath + f'{cond_on}_counts.png')
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
        slf_cm = self.get_cm(self.slf_test, preds[1])
        slf_labs = labels['selfing']
        self.plot_cm(slf_cm, slf_labs, 'selfing')

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


##############
#   MODELS   #
##############
class OneChannelClassifier(BaseClassifier):
    def __init__(self, balancing='keep', mutation='keep', one_channel=True):
        super().__init__('one_channel', balancing, mutation, one_channel)

    def compile_model(self):
        # build network
        x = Input(self.x_train[0,:].shape)

        y = Conv2D(6, (4, 15), activation='relu')(x)
        y = Dropout(0.4)(y)
        y = Conv2D(12, (4, 20), activation='relu')(y)
        y = Dropout(0.4)(y)
        y = Conv2D(24, (4, 15), activation='relu')(y)
        y = Dropout(0.4)(y)
        y = Conv2D(48, (2, 15), activation='relu')(y)
        y = Flatten()(y)
        y = Dense(1000, 'relu')(y)
        y = Dropout(0.4)(y)
        y = Dense(200, 'relu')(y)
        y = Dropout(0.4)(y)
        y = Dense(20, 'relu')(y)

        # concatenate last hidden layers
        mut_pred = Dense(self.num_mut_labs, 'softmax', name='mutation')(y)
        slf_pred = Dense(self.num_slf_labs, 'softmax', name='selfing')(y)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(y)

        model = Model(x, [mut_pred, slf_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        optmzr = Adam(learning_rate=0.0001)
        model.compile(optmzr, losses, loss_weights=[1.0, 0.5, 1.0],
                      metrics=['accuracy'])
        return model


class Conv3DClassifier(BaseClassifier):
    def __init__(self, balancing='keep', mutation='keep'):
        super().__init__('conv3d', balancing, mutation)
        self.expand_fvec_dims()

    def expand_fvec_dims(self):
        """ Adds a dimension to fvecs for Conv3D layers """
        self.x_train = np.expand_dims(self.x_train, 4)
        self.x_test = np.expand_dims(self.x_test, 4)
        assert self.x_train.shape[1:] == (13, 25, 3, 1)

    def compile_model(self):
        # build network
        x = Input(self.x_train[0,:].shape)

        y = Conv3D(6, (4, 10, 2), activation='relu')(x)
        y = Dropout(0.4)(y)
        y = Conv3D(12, (4, 4, 1), activation='relu')(y)
        y = Dropout(0.4)(y)
        y = Conv3D(24, (4, 4, 1), activation='relu')(y)
        y = Dropout(0.4)(y)
        y = Conv3D(48, (4, 4, 1), activation='relu')(y)
        y = Flatten()(y)
        y = Dense(1000, 'relu')(y)
        y = Dropout(0.4)(y)
        y = Dense(200, 'relu')(y)
        y = Dropout(0.4)(y)
        y = Dense(20, 'relu')(y)

        # concatenate last hidden layers
        mut_pred = Dense(self.num_mut_labs, 'softmax', name='mutation')(y)
        slf_pred = Dense(self.num_slf_labs, 'softmax', name='selfing')(y)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(y)

        model = Model(x, [mut_pred, slf_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        optmzr = Adam(learning_rate=0.0001)
        model.compile(optmzr, losses, loss_weights=[1.0, 0.5, 1.0],
                      metrics=['accuracy'])
        return model

    
class VanillaClassifier(BaseClassifier):
    def __init__(self, balancing='keep', mutation='keep', one_channel=False):
        super().__init__('vanilla', balancing, mutation, one_channel)

    def compile_model(self):
        # build network
        x = Input(self.x_train[0,:].shape)

        y = Conv2D(6, (4, 3), activation='relu')(x)
        y = Dropout(0.3)(y)
        y = Conv2D(12, (4, 5), activation='relu')(y)
        y = Dropout(0.3)(y)
        y = Conv2D(24, (4, 5), activation='relu')(y)
        y = Dropout(0.3)(y)
        y = Conv2D(48, (2, 4), activation='relu')(y)
        y = Flatten()(y)
        y = Dense(100, 'relu')(y)
        y = Dropout(0.3)(y)
        y = Dense(20, 'relu')(y)

        # concatenate last hidden layers
        mut_pred = Dense(self.num_mut_labs, 'softmax', name='mutation')(y)
        slf_pred = Dense(self.num_slf_labs, 'softmax', name='selfing')(y)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(y)

        model = Model(x, [mut_pred, slf_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        optmzr = Adam(learning_rate=0.0001)
        model.compile(optmzr, losses, loss_weights=[1.0, 1.0, 1.0],
                      metrics=['accuracy'])
        return model


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
        slf_pred = Dense(self.num_slf_labs, 'softmax', name='selfing')(mss)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(mss)

        model = Model(x, [mut_pred, slf_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        model.compile('adam', losses, loss_weights=[1.0, 1.0, 1.0],
                      metrics=['accuracy'])
        return model


class TowerClassifier(BaseClassifier):
    def __init__(self, balancing='keep', mutation='keep'):
        super().__init__('tower', balancing, mutation)

    def compile_model(self):
        # build network
        x = Input(self.x_train[0,:].shape)

        def build_tower(x):
            y = Conv2D(6, (2, 5), activation='relu')(x)
            y = Dropout(0.3)(y)
            y = Conv2D(12, (3, 5), activation='relu')(y)
            y = Dropout(0.3)(y)
            y = Conv2D(24, (4, 5), activation='relu')(y)
            y = Dropout(0.3)(y)
            y = Conv2D(48, (5, 5), activation='relu')(y)
            y = Flatten()(y)
            return y

        def build_branch(x):
            y = Dense(100, 'relu')(x)
            y = Dropout(0.3)(y)
            y = Dense(20, 'relu')(y)
            return y

        def build_network(x):
            # towers
            towers = [build_tower(x) for _ in range(3)]

            # tower branches
            branches = [[build_branch(t) for _ in range(3)] for t in towers]

            # connect to losses
            mut, slf, sel = [], [], []
            for b in branches: # loops through mut, slf, sel tower branches
                mut.append(b[0])
                slf.append(b[1])
                sel.append(b[2])

            # concatenate last layers
            muts = concatenate(mut)
            slfs = concatenate(slf)
            sels = concatenate(sel)
            mut_pred = Dense(self.num_mut_labs,
                             'softmax',
                             name='mutation')(muts)
            slf_pred = Dense(self.num_slf_labs,
                             'softmax',
                             name='selfing')(slfs)
            sel_pred = Dense(self.num_sel_labs,
                             'softmax',
                             name='selection')(sels)

            return Model(x, [mut_pred, slf_pred, sel_pred])

        model = build_network(x)

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        model.compile('adam', losses, loss_weights=[0.5, 0.8, 1.0],
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
        slf_pred = Dense(self.num_slf_labs, 'softmax', name='selfing')(y)
        sel_pred = Dense(self.num_sel_labs, 'softmax', name='selection')(y)

        model = Model(x, [mut_pred, slf_pred, sel_pred])

        # losses
        losses = [CategoricalCrossentropy(from_logits=True), 
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]

        # loss weights for old data: [0.2, 1.0, 1.0]
        model.compile('adam', losses, loss_weights=[1.0, 1.0, 1.0],
                      metrics=['accuracy'])
        return model
