import numpy as np
import tensorflow as tf


class SynthMathData(tf.keras.utils.Sequence):
    """
    - generates minibatches of batch_size samples
    - each sample x is a sequence of length seq_len, with each feature consisting of 2 real numbers (a,b):
        - a is a uniform random number in the range 0...1
        - b is zero, except for 2 randomly chosen features in the first half of the sequence where it is 1
    - target y is
        - average (a1 + a2)/2 if operation=="Add", where a1 and a2 are from the features where b==1
        - a1 * a2 if operation=="Multiply"
    - this is very similar to Experiments 4 and 5 from the 1997 Hochreiter/Schmidhuber LSTM paper
    """

    def __init__(self, batch_size, seq_len, operation, distribution, seed=42):

        assert operation in ["add", "multiply"]
        assert distribution in ["uniform", "normal"]

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.operation = operation
        self.distribution = distribution
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        # 1 epoch consists of 1 minibatch

        return 1

    def __getitem__(self, idx):
        # idx is irrelevant but required

        x_batch, y_batch = [], []
        for _ in range(self.batch_size):
            if self.distribution == "uniform":
                upper = self.rng.uniform(0, 1, self.seq_len)
            elif self.distribution == "normal":
                upper = self.rng.normal(0, 1, self.seq_len)
            x = np.transpose(np.stack([ upper, np.zeros(self.seq_len) ]))
            id1, id2 = self.rng.choice(self.seq_len // 2, 2, replace=False)
            x[id1,1], x[id2,1] = 1, 1
            x1, x2 = x[id1,0], x[id2,0]
            if self.operation == "add":
                y = (x1 + x2) / 2.0
            elif self.operation == "multiply":
                y = x1 * x2
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch).astype(np.float32), np.array(y_batch).astype(np.float32)
    


class SynthCosSimData(tf.keras.utils.Sequence):


    def __init__(self, batch_size, seq_len, emb_dim, noise_factor=1.0, seed=42):
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.noise_factor = noise_factor
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        
        return 1

    def __getitem__(self, idx):

        x_batch, y_batch = [], []
        for _ in range(self.batch_size):
            x = self.rng.normal(0, 1, (self.seq_len, self.emb_dim))
            if self.rng.uniform() > 0.5:
                idx1, idx2 = self.rng.choice(self.seq_len, 2, replace=False)
                #idx2 += self.seq_len // 2
                x[idx2] = x[idx1]
                if self.noise_factor > 0:
                    x[idx2] += self.rng.normal(0, self.noise_factor, self.emb_dim)
                y = 1
            else:
                y = 0
            x /= np.linalg.norm(x, axis=-1, keepdims=True)
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch).astype(np.float32), np.array(y_batch).astype(np.int8)