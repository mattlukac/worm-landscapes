from classifiers import *
import numpy as np

# model building
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adadelta, Adam, Adagrad, Nadam
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tensorflow.io.gfile import glob, exists, mkdir, remove
from tensorflow.random import set_seed

# data wrangling
import pickle
import copy
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class MetaClassifier(BaseClassifier):
    def __init__(self, model_dirs, balancing='remove', mutation='keep'):
        super().__init__('meta', balancing, mutation)
        self.model_dirs = model_dirs

    def compile_model(self):
        """ 
        Loads pretrained models, attaches dense network
        to learn labels with pretrained softmaxes as features
        """
        # load and rename pretrained models
        model1, model2, model3 = self.load_pretrained_models()
        model1._name = 'conv2d_3channel'
        model2._name = 'conv2d_1channel'
        model3._name = 'conv3d_1channel'

        # assign inputs
        x1 = Input(self.x1_train[0,:].shape)
        x2 = Input(self.x2_train[0,:].shape)
        x3 = Input(self.x3_train[0,:].shape)
        inputs = [x1, x2, x3]

        # pretrained model outputs, each a list of softmax
        # activations with [mutation, selfing, selection]
        y1 = concatenate( model1(x1, training=False) )
        y2 = concatenate( model2(x2, training=False) )
        y3 = concatenate( model3(x3, training=False) )
        y = concatenate([y1, y2, y3])
        
        # attach dense network to end of pretrained models
        y = Dense(100, 'relu')(y)
        y = Dropout(0.2)(y)
        mut = Dense(self.num_mut_labs, 'softmax', name='mutation')(y)
        slf = Dense(self.num_slf_labs, 'softmax', name='selfing')(y)
        sel = Dense(self.num_sel_labs, 'softmax', name='selection')(y)
        outputs = [mut, slf, sel]

        # losses, optimizer
        losses = [CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True),
                  CategoricalCrossentropy(from_logits=True)]
        optmzr = Adam(learning_rate=0.0001)

        # compile transfer learn classifier
        model = Model(inputs, outputs)
        model.compile(optmzr, losses, loss_weights=[1.0, 0.5, 1.0],
                      metrics=['accuracy'])
        return model

    def load_pretrained_models(self):
        """
        Given model name and version numbers,
        load model checkpoints and make them non-trainable
        """
        models = []
        for md in self.model_dirs:
            model_path = f'model_logs/{md}'
            ckpt = glob(model_path + '*hdf5')[-1]
            print(f'loading model from {md + ckpt}')
            model = load_model(ckpt)
            model.trainable = False
            models.append(model)

        return models

    def stack_3d_fvecs(self, fvecs):
        # prep 1 channel conv3d fvecs
        fvecs3 = copy.deepcopy(fvecs)
        fvecs3 = np.expand_dims(fvecs3, 4)
        return fvecs3

    def prep_data(self, balancing, mutation, unstack=None):
        """
        Loads and splits feature vectors for each pretrained network
        """
        data = self.load_fvecs_and_targets()

        # remove balancing sel from data
        if balancing == 'remove':
            data = self.remove_balancing(data)
        # remove 2-1-2 mutation from data
        if mutation == 'remove':
            data = self.remove_2x_mutation(data)

        # unpack data
        fvecs, mutation, selfing, selection = data

        # prep 3 channel, 1 channel, 3d fvecs
        fvecs1 = copy.deepcopy(fvecs)
        fvecs2 = self.unstack_fvecs(fvecs)
        fvecs3 = self.stack_3d_fvecs(fvecs)

        # count number of labels
        self.num_mut_labs = mutation.shape[1]
        self.num_slf_labs = selfing.shape[1]
        self.num_sel_labs = selection.shape[1]

        # split to train and test sets
        split_data = train_test_split(fvecs1, fvecs2, fvecs3, 
                                 mutation, selfing, selection,
                                 test_size=0.3, random_state=23)
        if balancing == 'resamp':
            split_data = self.resample_balancing(split_data)

        (self.x1_train, self.x1_test, 
         self.x2_train, self.x2_test,
         self.x3_train, self.x3_test,
         self.mut_train, self.mut_test,
         self.slf_train, self.slf_test,
         self.sel_train, self.sel_test) = split_data

    def fit_model(self, max_epochs, verbose=False):
        """ Train transfer learning classifier """
        if self.model is None:
            self.model = self.compile_model()
        else:
            self.model = self.load_best_model()

        if verbose: self.model.summary()

        # callbacks
        callbacks = self.get_callbacks(max_epochs)

        # train network
        x_train = [self.x1_train, self.x2_train, self.x3_train]
        x_test = [self.x1_test, self.x2_test, self.x3_test]
        y_train = [self.mut_train, self.slf_train, self.sel_train]
        y_test = [self.mut_test, self.slf_test, self.sel_test]
        valid_data = (x_test, y_test)
        set_seed(23)
        self.history = self.model.fit(x_train, y_train,
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

    def predict(self, test_data):
        """ 
        Predict test set data on trained transfer learning classifier 
        """
        assert self.is_trained

        # load empirical data
        data1 = self.load_data(filename)
        data2 = self.fvec_to_conv2d(data1)
        data3 = self.fvec_to_conv3d(data1)
        inputs = [data1, data2, data3]
        preds = self.model.predict(inputs)
        return preds


if __name__ == "__main__":
    model_dirs = ['vanilla/version_88/', 
                  'vanilla/version_117/', 
                  'conv3d/version_7/']
    meta = MetaClassifier(model_dirs)
    meta.fit_model(max_epochs=10, verbose=True)
