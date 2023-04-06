import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.activations import relu, gelu
from tensorflow.keras.regularizers import L1, L2


def build_model_mlp_2d_01(
        input_shape,
        layer_units,
):    
    assert isinstance(layer_units, list)
    assert len(input_shape) in [2,3]

    inp = Input(shape=input_shape, name="input")
    x = inp

    if len(input_shape) == 2:
        x = tf.expand_dims(x, axis=-1) # required for Conv2D

    for layer, units in enumerate(layer_units):
        if layer == 0:
            x = Conv2D(units, input_shape[:2], name=f"dense_{layer}")(x)
            x = Flatten(name=f"flatten_{layer}")(x)
        else:
            x = relu(x)
            x = Dense(units, name=f"dense_{layer}")(x)

    out = x
    model = Model(inp, out, name="mlp_2d_01")

    return model