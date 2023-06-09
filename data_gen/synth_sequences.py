import numpy as np
import tensorflow as tf


class SynthNNumbers(tf.keras.utils.Sequence):

    def __init__(self, batch_size, n, operation, limit, distribution, seed=42):

        assert operation in ["plus", "multiply"]
        assert distribution in ["uniform", "normal"]

        self.batch_size = batch_size
        self.n = n
        self.operation = operation
        self.limit = limit
        self.distribution = distribution
        self.rng = np.random.default_rng(seed=seed)

    def __len__(self):
        # 1 epoch consists of 1 minibatch

        return 1

    def __getitem__(self, idx):
        # idx is irrelevant but required

        if self.distribution == "uniform":
            x_batch = self.rng.uniform(-self.limit, self.limit, (self.batch_size, self.n))
        elif self.distribution == "normal":
            x_batch = self.rng.normal(0, self.limit, (self.batch_size, self.n))

        if self.operation == "plus":
            y_batch = np.sum(x_batch, axis=-1)
        elif self.operation == "multiply":
            y_batch = np.prod(x_batch, axis=-1)
        return x_batch.astype(np.float32), y_batch.astype(np.float32)



class SynthMathData(tf.keras.utils.Sequence):
    """
    - generates minibatches of batch_size samples
    - each sample x is a sequence of length seq_len, with each feature consisting of 2 real numbers (a,b):
        - a is a uniform random number in the range 0...1
        - b is zero, except for 2 randomly chosen features in the first half of the sequence where it is 1
    - target y is
        - "renormalized" sum (a1 + a2)/2 if operation=="plus", where a1 and a2 are from the features where b==1
        - "renormalized" difference (a1 - a2)/2 if operation=="minus"
        - a1 * a2 if operation=="multiply"
    - this is very similar to Experiments 4 and 5 from the 1997 Hochreiter/Schmidhuber LSTM paper
    """

    def __init__(self, batch_size, seq_len, operation, distribution, seed=42):

        assert operation in ["plus", "minus", "multiply"]
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
                upper = self.rng.uniform(-1, 1, self.seq_len)
            elif self.distribution == "normal":
                upper = self.rng.normal(0, 1, self.seq_len)
            x = np.transpose(np.stack([ upper, np.zeros(self.seq_len) ]))
            id1, id2 = self.rng.choice(self.seq_len // 2, 2, replace=False)
            x[id1,1], x[id2,1] = 1, 1
            x1, x2 = x[id1,0], x[id2,0]
            if self.operation == "plus":
                y = (x1 + x2) / 2.0 # keep result in range -1...1 if distribution=="uniform"
            elif self.operation == "minus":
                y = (x1 - x2) / 2.0 # keep result in range -1...1 if distribution=="uniform"
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