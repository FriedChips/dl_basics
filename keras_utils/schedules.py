import numpy as np
#import pandas as pd
#import json
#import os
import tensorflow as tf
import matplotlib.pyplot as plt
#import random as python_random



class ExpCosSegments:
    '''
    Keras schedule, see https://keras.io/api/callbacks/learning_rate_scheduler/
    The exponent of the learning rate varies in segments like a half-period cosine.
    segment_list : list of segments, each of the form [ exp_start, exp_end, length ]
    exp_start : lr exponent to base 10 at segment start,
    exp_end   : lr exponent to base 10 at segment end,
    length    : segment lenght in epochs
    '''

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
        ''' visualize/check the schedule with a log plot '''

        fig, ax = plt.subplots(1,1, figsize=(5,2))
        ax.plot(self.schedule)
        ax.grid(True)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        plt.show()


if __name__ == "__main__":
    ExpCosSegments([[-7,-3,128], [-3,-3,256], [-3,-5,128]]).plot_schedule()
