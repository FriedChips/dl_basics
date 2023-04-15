import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt



class Schedule_ExpCos_Segments:
    ''' Keras schedule '''

    def __init__(self, segment_list):
        
        self.schedule = []
        
        for segment in segment_list:
            exp_start, exp_end, length = segment
            cos_ampl = (exp_start - exp_end) / 2
            cos_offset = exp_start - cos_ampl
            exp = cos_ampl * np.cos(np.pi * np.arange(length) / length) + cos_offset
            self.schedule.extend(10 ** exp)
            
        self.schedule = np.array(self.schedule)
        self.len_schedule = len(self.schedule)

        
    def scheduler(self, epoch, lr):
    
        if epoch < self.len_schedule:
            return self.schedule[epoch]
        else: # fallback if training continues longer than schedule length
            return self.schedule[-1]


    def plot_schedule(self):

        fig, ax = plt.subplots(1,1, figsize=(5,2))
        ax.plot(self.schedule);
        ax.set_yscale("log")
        ax.set_xlabel("epoch");
        ax.set_ylabel("learning rate");



def max_abs_error(y_true, y_pred):
    ''' metric which returns the largest absolute prediction error '''

    return tf.reduce_max(tf.abs(y_true - y_pred), axis=-1)



class TrainingRun:

    def __init__(self):

        self.history = None


    def update_history(self, history):
        
        self.history = pd.concat([self.history, pd.DataFrame(history.history)], ignore_index=True)
        self.history["epoch"] = self.history.index + 1
        self.history = self.history.set_index("epoch")


    def save_state(self, directory, P, model):

        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "P.json"), "w") as f:
            json.dump(P, f, indent=3)
        if self.history is not None:
            self.history.to_csv(os.path.join(directory, "train_history.csv"), index=True)
            current_epoch = len(self.history)
        else:
            current_epoch = 0
        model.save(os.path.join(directory, f"model-epoch-{current_epoch:06d}.hdf5"))


    def load_state(self, directory):

        with open(os.path.join(directory, "P.json"), "r") as f:
            P = json.load(f)
        self.history = pd.read_csv(os.path.join(directory, "train_history.csv"))
        current_epoch = len(self.history)
        model = tf.keras.models.load_model(os.path.join(directory, f"model-epoch-{current_epoch:06d}.hdf5"))
        return P, model



class SaveWeightsPower2(tf.keras.callbacks.Callback):
    
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def on_train_begin(self, logs=None):
        self.exponent = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 2**self.exponent - 1: # keras internally counts epochs starting from 0
            self.model.save_weights(os.path.join(directory, f"weights-epoch-{epoch+1:06d}.hdf5"))
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
