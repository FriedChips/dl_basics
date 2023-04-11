import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Embedding
from tensorflow.keras.activations import relu, gelu
from tensorflow.keras.regularizers import L1, L2


def transformer_block(inp, num_heads, head_dim=None, mlp_factor=4, layer_norm=False, mlp_act="relu"):

    assert mlp_act in [ "relu", "gelu" ]
    embedding_dim = inp.shape[-1]

    if head_dim is None:
        head_dim = embedding_dim // num_heads

    x = inp
    if layer_norm:
        x = LayerNormalization(epsilon=1e-3)(x)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)
    res = x + inp

    x = res
    if layer_norm:
        x = LayerNormalization(epsilon=1e-3)(x)
    x = Dense(embedding_dim * mlp_factor)(x)
    x = eval(mlp_act)(x)
    x = Dense(embedding_dim)(x)
    out = x + res

    return out


def transformer_01(
        input_shape,
        num_heads,
        num_tokens,
        embedding_dim=None,
        return_final_token_only=False,
):
    assert len(input_shape) in [1,2]

    inp = Input(shape=input_shape, name="input")
    x = inp

    if len(input_shape) == 1:
        x = Embedding(num_tokens, embedding_dim)(x)

    x = transformer_block(x, num_heads=num_heads)
    x = Dense(num_tokens)(x)

    if return_final_token_only:
        x = x[..., -1, :]
    out = x

    model = Model(inp, out)

    return model


if __name__ == "__main__":

    # building test
    transformer_01(
        input_shape=(64,),
        num_heads=4,
        num_tokens=32,
        embedding_dim=128
    ).summary()