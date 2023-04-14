import tensorflow as tf
import numpy as np
import pandas as pd
import os


class Quadratic(tf.keras.layers.Layer):
    '''
    TODO:
    - add config for serializing
    - rework weights to conform to keras standard (transpose them)
    '''
    
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )
        self.w = self.add_weight(
            shape=(self.units, input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.u = self.add_weight(
            shape=(self.units, input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, x):
        y = self.b
        y = y + tf.einsum("...ij,...j -> ...i", self.w, x)
        y = y + tf.einsum("...ijk,...j,...k ->...i", self.u, x, x)
        return y



class Laplace(tf.keras.regularizers.Regularizer):

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        loss = tf.reduce_sum((
            tf.roll(x, shift=+1, axis=0) +
            tf.roll(x, shift=-1, axis=0) +
            tf.roll(x, shift=+1, axis=1) +
            tf.roll(x, shift=-1, axis=1) -
            4 * x) ** 2) * self.alpha
        return loss

    def get_config(self):
        return {'alpha': self.alpha}



class SaveWeightsPower2(tf.keras.callbacks.Callback):
    
    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id

    def on_train_begin(self, logs=None):
        self.exponent = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 2**self.exponent - 1: # keras internally counts epochs starting from 0
            self.model.save_weights(os.path.join(RUN_DIR, f"run{self.run_id:03d}-weights-e{epoch+1:05d}.hdf5"))
            self.exponent += 1



class LogWeightNorms(tf.keras.callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.reset_norms()

    def on_epoch_end(self, epoch, logs=None):
        self.weight_norms.append(self.calc_norms())

    def calc_norms(self):
        return [ np.linalg.norm(w.reshape(-1)) for w in self.model.get_weights() ]

    def reset_norms(self):
        self.weight_norms = []

    def norms_dataframe(self):
        num_cols = len(self.weight_norms[0])
        num_rows = len(self.weight_norms)
        col_names = [ f"w_norm_{c:02d}" for c in range(num_cols) ]
        df = pd.DataFrame(self.weight_norms, columns=col_names, index=range(1, num_rows+1))
        df.index.name = "epoch"
        return df
    
    def norms_to_csv(self, directory):
        self.norms_dataframe().to_csv(os.path.join(directory, "weight_norms.csv"), index=True)


