import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.activations import relu, gelu
from tensorflow.keras.regularizers import L1, L2


def mlp_01(
        input_shape,
        layer_units,
        activations,
):    
    assert isinstance(layer_units, list) and isinstance(activations, list)
    assert len(input_shape) <= 3
    assert len(layer_units) == len(activations)

    inp = Input(shape=input_shape, name="input")
    x = inp

    if len(input_shape) == 2:
        x = tf.expand_dims(x, axis=-1) # required for Conv2D

    for layer, (units, act) in enumerate(zip(layer_units, activations)):
        if (layer == 0) and len(input_shape) > 1:
            x = Conv2D(units, input_shape[:2], name=f"dense_{layer}")(x)
            x = Flatten(name="flatten")(x)
        else:
            x = Dense(units, name=f"dense_{layer}")(x)
        assert act in ["linear", "relu", "gelu"]
        if act is not "linear":
            x = eval(act)(x)

    out = x
    model = Model(inp, out, name="mlp_01")

    return model


if __name__ == "__main__":

    # building test
    mlp_01(
        input_shape=(28,28),
        layer_units=[32,16,8],
        activations=["relu", "linear", "gelu"],
    ).summary()