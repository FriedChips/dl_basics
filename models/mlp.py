import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.regularizers import L1, L2


def mlp_01(
        input_shape,
        layer_units,
        activations,
        use_bias=True,
        regularizer=None,
):    
    assert isinstance(layer_units, list) and isinstance(activations, list)
    assert len(input_shape) <= 3
    assert len(layer_units) == len(activations)

    if regularizer is not None:
        regularizer = eval(regularizer)

    inp = Input(shape=input_shape, name="input")
    x = inp

    if len(input_shape) == 2:
        x = tf.expand_dims(x, axis=-1) # required for Conv2D

    for layer, (units, act) in enumerate(zip(layer_units, activations)):
        if (layer == 0) and len(input_shape) > 1:
            x = Conv2D(
                units, input_shape[:2],
                use_bias=use_bias,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=f"dense_{layer}"
            )(x)
            x = Flatten(name="flatten")(x)
        else:
            x = Dense(
                units,
                use_bias=use_bias,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=f"dense_{layer}"
            )(x)
        if act != "linear":
            x = eval("tf.keras.activations." + act)(x)

    out = x
    model = Model(inp, out, name="mlp_01")

    return model


def mlp_residual(
        input_shape,
        layer_units,
        activations,
        use_bias=True,
        regularizer=None,
):    
    assert isinstance(layer_units, list) and isinstance(activations, list)
    assert len(layer_units) == len(activations)
    assert len(input_shape) == 1

    if regularizer is not None:
        regularizer = eval(regularizer)

    inp = Input(shape=input_shape, name="input")
    x = inp

    for layer, (units, act) in enumerate(zip(layer_units, activations)):

        x_resid = Dense(
            units,
            use_bias=use_bias,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            name=f"dense_residual_{layer}"
        )(x)

        if act != "linear":

            x_nonlin = Dense(
                units,
                use_bias=use_bias,
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer,
                name=f"dense_nonlin_{layer}"
            )(x)
            x_nonlin = eval("tf.keras.activations." + act)(x_nonlin)

            x = x_nonlin + x_resid

        else:

            x = x_resid

    out = x
    model = Model(inp, out, name="mlp_residual")

    return model


if __name__ == "__main__":

    # building test
    '''
    mlp_01(
        input_shape=(28,28),
        layer_units=[32,16,8,4],
        activations=["relu", "linear", "gelu", "tanh"],
        use_bias=False,
    ).summary()
    '''

    mlp_residual(
        input_shape=(28*28,),
        layer_units=[256,256,10],
        activations=["linear", "relu", "linear"],
        use_bias=False,
    ).summary()