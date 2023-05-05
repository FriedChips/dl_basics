import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
#import matplotlib.pyplot as plt
import random as python_random



class TrainingRun:

    def __init__(self):

        self.history = None
        self.current_epoch = 0


    def update_history(self, history):
        
        self.history = pd.concat([self.history, pd.DataFrame(history)], ignore_index=True)
        self.history["epoch"] = self.history.index + 1
        self.history = self.history.set_index("epoch")
        self.current_epoch = len(self.history)


    def save_state(self, directory, P, model):

        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "P.json"), "w") as f:
            json.dump(P, f, indent=3)
        if self.history is not None:
            self.history.to_csv(os.path.join(directory, "train_history.csv"), index=True)
        model.save(os.path.join(directory, f"model-epoch-{self.current_epoch:06d}.hdf5"))


    def load_state(self, directory):

        with open(os.path.join(directory, "P.json"), "r") as f:
            P = json.load(f)
        self.history = pd.read_csv(os.path.join(directory, "train_history.csv"))
        self.current_epoch = len(self.history)
        model = tf.keras.models.load_model(os.path.join(directory, f"model-epoch-{self.current_epoch:06d}.hdf5"))
        return P, model



def tf_keras_random_seed(seed):
    ''' code from https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development '''

    tf.keras.backend.clear_session()

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(seed)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(seed)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(seed)

