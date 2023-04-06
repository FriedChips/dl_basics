from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Flatten
from tensorflow.keras.regularizers import L1, L2


def build_model_gru_01(
        input_shape,
        num_outputs,
        gru_units,
        num_hidden_gru = 0,
        return_final_hidden = False,
        use_embedding = False,
        vocab_size = None,
        embedding_dim = None,
):    
    inp = Input(shape=input_shape, name="input")
    x = inp

    if use_embedding:
        x = Embedding(vocab_size, embedding_dim, input_length=input_shape[0], name="embedding")(x)
    for i in range(num_hidden_gru):
        x = GRU(gru_units, return_sequences=True, name=f"gru_{i}")(x)
    x = GRU(gru_units, return_sequences=return_final_hidden, name="gru_final")(x)
    if return_final_hidden:
        x = Flatten(name="flatten")(x)
    x = Dense(num_outputs, name="linear_final")(x)

    out = x
    model = Model(inp, out, name="gru_01")

    return model