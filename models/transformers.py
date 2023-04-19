import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Embedding
from tensorflow.keras.activations import relu, gelu
from tensorflow.keras.regularizers import L1, L2


def transformer_block(
        input, # shape (sequence_length, embedding_dim)
        num_heads, # number of attention heads
        head_dim=None, # width of each head. if None, then set automatically, see below
        use_mlp=True, # include 2-layer MLP after the attention layer?
        mlp_factor=4, # width of intermediate MLP layer as a mutiple of embedding_dim
        mlp_act="relu",
        layer_norm=False, # use layer normalization?
        layer_norm_eps = 1e-6,
        use_bias=True,
):

    assert mlp_act in [ "relu", "gelu" ]
    embedding_dim = input.shape[-1]

    if head_dim is None:
        head_dim = embedding_dim // num_heads

    x = input
    if layer_norm:
        x = LayerNormalization(epsilon=layer_norm_eps)(x)
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim, use_bias=use_bias)(x, x)
    res = x + input

    if use_mlp:
        x = res
        if layer_norm:
            x = LayerNormalization(epsilon=layer_norm_eps)(x)
        x = Dense(embedding_dim * mlp_factor, activation=mlp_act, use_bias=use_bias)(x)
        x = Dense(embedding_dim, use_bias=use_bias)(x)
        out = x + res
    else:
        out = res

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