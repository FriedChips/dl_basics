import numpy as np
from sklearn.model_selection import train_test_split


def addition_modulo_data(p_max, train_size, add_equal_sign=False, seed=42):

    x_full = np.array([ [i, j, p_max] for i in range(p_max) for j in range(p_max) ]).astype(np.int32)
    y_full = (np.sum(x_full[:, :-1], axis=-1) % p_max).astype(np.int32)

    if not add_equal_sign:
        x_full = x_full[:, :-1]

    x_train, x_val, y_train, y_val = train_test_split(x_full, y_full, train_size=train_size, random_state=seed)
    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = addition_modulo_data(113, 0.3, add_equal_sign=True)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    print(x_train.dtype, x_val.dtype, y_train.dtype, y_val.dtype)
    print(x_train[:3], y_train[:3], "\n", x_val[:3], y_val[:3])
