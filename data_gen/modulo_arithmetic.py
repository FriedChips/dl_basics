import numpy as np
from sklearn.model_selection import train_test_split


def addition_modulo_data(p_max, train_size, seed=42):

    x_full = np.array([ [i,j] for i in range(p_max) for j in range(p_max) ]).astype(np.int32)
    y_full = (np.sum(x_full, axis=-1) % p_max).astype(np.int32)

    x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, train_size=train_size, random_state=seed)
    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = addition_modulo_data(113, 0.3)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    print(x_train.dtype, x_val.dtype, y_train.dtype, y_val.dtype)
    print(x_train[:3], y_train[:3], x_val[:3], y_val[:3])
