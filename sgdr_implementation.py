import keras
from keras.callbacks import Callback
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras import optimizers
import math
import random
import numpy as np
import matplotlib.pyplot as plt


class LR_Updater(Callback):
    '''This callback is utilized to log learning rates every iteration (batch cycle)
    it is not meant to be directly used as a callback but extended by other callbacks
    ie. LR_Cycle
    '''
    def __init__(self, iterations):
        '''
        iterations = dataset size / batch size
        epochs = pass through full training dataset
        '''
        self.epoch_iterations = iterations
        self.trn_iterations = 0.
        self.history = {}
    def on_train_begin(self, logs={}):
        self.trn_iterations = 0.
        logs = logs or {}
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        K.set_value(self.model.optimizer.lr, self.setRate())
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
    def plot_lr(self):
        plt.xlabel("iterations")
        plt.ylabel("learning rate")
        plt.plot(self.history['iterations'], self.history['lr'])
    def plot(self, n_skip=10):
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')

        
class LR_Cycle(LR_Updater):
    '''This callback is utilized to implement cyclical learning rates
    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai
    and adopted from this keras implementation https://github.com/bckenstler/CLR
    '''
    def __init__(self, iterations, cycle_mult = 1):
        '''
        iterations = dataset size / batch size
        iterations = number of iterations in one annealing cycle
        cycle_mult = used to increase the cycle length cycle_mult times after every cycle
        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$
        '''
        self.min_lr = 0
        self.cycle_mult = cycle_mult
        self.cycle_iterations = 0.
        super().__init__(iterations)
    def setRate(self):
        self.cycle_iterations += 1
        if self.cycle_iterations == self.epoch_iterations:
            self.cycle_iterations = 0.
            self.epoch_iterations *= self.cycle_mult
        cos_out = np.cos(np.pi*(self.cycle_iterations)/self.epoch_iterations) + 1
        return self.max_lr / 2 * cos_out
    def on_train_begin(self, logs={}):
        super().on_train_begin(logs={}) #changed to {} to fix plots after going from 1 to mult. lr
        self.cycle_iterations = 0.
        self.max_lr = K.get_value(self.model.optimizer.lr)
