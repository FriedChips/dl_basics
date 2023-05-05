import numpy as np
#import pandas as pd
#import json
import os
import tensorflow as tf
#import matplotlib.pyplot as plt



class SaveWeightsPower2(tf.keras.callbacks.Callback):
    '''
    Keras callback, see https://keras.io/guides/writing_your_own_callbacks/
    Saves model weights after every epoch which is a power of 2 and also every save_interval epochs
    '''

    def __init__(self, directory, save_interval=2**20):
        super().__init__()
        self.directory = directory
        self.save_interval = save_interval # the default 2**20 is intentionally larger than any realistic epoch number

    def on_train_begin(self, logs=None):
        os.makedirs(self.directory, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        log_epoch = epoch + 1 # Keras internally counts epochs starting from 0
        if self.is_power_of_two(log_epoch) or (log_epoch % self.save_interval == 0):
            self.model.save_weights(os.path.join(self.directory, f"weights-epoch-{log_epoch:06d}.hdf5"))

    def is_power_of_two(self, n):
        ''' https://stackoverflow.com/questions/57025836 '''
        return (n & (n-1) == 0) and (n != 0)



class LogWeightNorms(tf.keras.callbacks.Callback):
    '''
    Keras callback, see https://keras.io/guides/writing_your_own_callbacks/
    ToDo: load norms from previous training, documentation
    '''

    def on_train_begin(self, logs=None):
        weights = self.model.get_weights()
        self.norm_scale = [ np.sqrt(len(w.reshape(-1))) for w in weights ]
        self.column_names = [ f"w_norm_{c:03d}" for c in range(len(weights)) ]
        self.history = {}
        for col_name in self.column_names:
            self.history[col_name] = []

    def on_epoch_end(self, epoch, logs=None):
        for col_name, norm in zip(self.column_names, self.calc_norms()):
            self.history[col_name].append(norm)

    def calc_norms(self):
        return [ np.linalg.norm(w.reshape(-1)) / s for w, s in zip(self.model.get_weights(), self.norm_scale) ]
    