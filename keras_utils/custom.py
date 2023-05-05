import tensorflow as tf
import numpy as np


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
    ''' ToDo: documentation'''
    
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
