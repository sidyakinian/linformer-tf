from json import decoder, encoder
import tensorflow as tf

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation

class EmbeddingsLayer(tf.keras.layers.Layer):
    def __init__(self, max_len: int, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=max_len, output_dim=d_model)
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Embeddings
        x = self.embedding(x)

        # Positional embeddings
        positions = tf.range(start=0, limit=self.max_len) # Use trig positions later, they ostensibly work better
        positions = self.position_embedding(positions)

        return x + positions


class MLP(tf.keras.layers.Layer):
    def __init__(self, d_ff: int, d_output: int, activation: Activation, dropout: float):
        super().__init__()
        self.fc1 = Dense(d_ff, activation=activation)
        self.fc2 = Dense(d_output)
        self.dropout = Dropout(dropout)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LinearSelfAttention(tf.keras.layers.Layer):
    def __init__(self, k: int, d_k: int, dropout: float):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.d_k = d_k
        self.e = Dense(k)
        self.f = Dense(k)

    def call(self, K: tf.Tensor, Q: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        # K, Q, V are all of shape (batch_size, n_heads, n, d_k)
        assert K.shape == Q.shape == V.shape, f"K, Q, V must have the same shape, but got {K.shape}, {Q.shape}, {V.shape}"
        K = tf.transpose(K, perm=[0, 1, 3, 2]) # (batch_size, n_heads, d_k, n)
        K = self.e(K) # (batch_size, n_heads, d_k, k)
        QK = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.d_k, dtype=tf.float32)) # shape (batch_size, n_heads, n, k)
        P_bar = tf.nn.softmax(QK, axis=-1) # along k dimension, otherwise pointless. shape (batch_size, n_heads, n, k)
        P_bar = self.dropout(P_bar)
        V = tf.transpose(V, perm=[0, 1, 3, 2]) # (batch_size, n_heads, d_k, n)
        V = self.f(V) # (batch_size, n_heads, d_k, k)
        V = tf.transpose(V, perm=[0, 1, 3, 2]) # (batch_size, n_heads, k, d_k)
        return tf.matmul(P_bar, V) # (batch_size, n_heads, n, d_k)

class MultiHeadLinearAttention(tf.keras.layers.Layer):
    # For MHA, just include n_heads as another dimension. Linear attention module should work with that dimension then
    def __init__(self, k: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.linear_attention = LinearSelfAttention(k=k, d_k=self.d_k, dropout=dropout)
        self.w_o = Dense(d_model)
        self.dropout = Dropout(dropout)

    def call(self, K: tf.Tensor, Q: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        # shape of K, Q, V: (batch_size, n, d_model)
        assert K.shape == Q.shape == V.shape, "K, Q, V must have the same shape"
        batch_size, n, d_k = K.shape

        def reshape_for_multihead_attention(M):
            M = tf.reshape(K, shape=[batch_size, n, self.n_heads, self.d_k])
            print(f"M.shape = {M.shape}")
            M = tf.transpose(M, perm=[0, 2, 1, 3]) # shape (batch_size, n_heads, n, d_k)
            return M

        K = reshape_for_multihead_attention(K)  
        Q = reshape_for_multihead_attention(Q)
        V = reshape_for_multihead_attention(V)
        
        attention_output = self.linear_attention(K, Q, V)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3]) # shape (batch_size, n, n_heads, d_k)
        attention_output = tf.reshape(attention_output, shape=[batch_size, n, self.d_model]) # shape (batch_size, n, d_model)
        # TODO: double check w_o
        outputs = self.w_o(attention_output)
    
        return outputs

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x

class Encoder(tf.keras.Model):
    def __init__(self, encoder_layer: EncoderLayer, n_layers: int):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.n_layers = n_layers

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for _ in range(self.n_layers):
            x = self.encoder_layer(x)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()


    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x

class Decoder(tf.keras.Model):
    def __init__(self, decoder_layer: DecoderLayer, n_layers: int):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.n_layers = n_layers

    def call(self, x: tf.Tensor, encoding: tf.Tensor) -> tf.Tensor:
        for _ in range(self.n_layers):
            x = self.decoder_layer(x, encoding)
        return x

class Linformer(tf.keras.Model):
    def __init__(self, embeddings_layer: tf.keras.layers.Layer, encoder: tf.keras.Model, decoder: tf.keras.Model):
        super().__init__()
        self.embeddings_layer = embeddings_layer
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Tokenize text (should be outside of this module probably)
        inputs = self.embeddings_layer(inputs)
        encoding = self.encoder(inputs)
        logits = self.decoder(inputs, encoding)
        # Softmax and ff on outputs
        outputs = logits # TODO: Change this
        return outputs

    def num_params(self) -> int:
        trainable_params = tf.math.reduce_sum(tf.size(v) for v in self.trainable_weights).numpy()
        non_trainable_params = tf.math.reduce_sum(tf.size(v) for v in self.non_trainable_weights).numpy()
        return trainable_params + non_trainable_params
