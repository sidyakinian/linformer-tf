from json import decoder, encoder
import tensorflow as tf

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation, GlobalAveragePooling1D
from utils import reshape_for_multihead_attention

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
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class LinearSelfAttention(tf.keras.layers.Layer):
    def __init__(self, k: int, d_k: int, dropout: float, full_attn: bool):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.d_k = d_k
        self.full_attn = full_attn
        if not full_attn:
            self.e = Dense(k)
            self.f = Dense(k)

    def call(self, K: tf.Tensor, Q: tf.Tensor, V: tf.Tensor, ) -> tf.Tensor:
        # K, Q, V are all of shape (batch_size, n_heads, n, d_k)
        # assert tf.shape(K) == tf.shape(Q) == tf.shape(V), "K, Q, V must have the same shape"
        K = tf.transpose(K, perm=[0, 1, 3, 2]) # (batch_size, n_heads, d_k, n)
        if not self.full_attn:
            K = self.e(K) # (batch_size, n_heads, d_k, k)
        QK = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.d_k, dtype=tf.float32)) # shape (batch_size, n_heads, n, k)
        P_bar = tf.nn.softmax(QK, axis=-1) # along k dimension, otherwise pointless. shape (batch_size, n_heads, n, k)
        P_bar = self.dropout(P_bar)
        V = tf.transpose(V, perm=[0, 1, 3, 2]) # (batch_size, n_heads, d_k, n)
        if not self.full_attn:
            V = self.f(V) # (batch_size, n_heads, d_k, k)
        V = tf.transpose(V, perm=[0, 1, 3, 2]) # (batch_size, n_heads, k, d_k)
        return tf.matmul(P_bar, V) # (batch_size, n_heads, n, d_k)

class MultiHeadLinearAttention(tf.keras.layers.Layer):
    def __init__(self, k: int, d_model: int, n_heads: int, dropout: float, full_attn: bool):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.linear_attention = LinearSelfAttention(k=k, d_k=self.d_k, dropout=dropout, full_attn=full_attn)
        self.w_o = Dense(d_model)
        self.dropout = Dropout(dropout)

    def call(self, K: tf.Tensor, Q: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        # shape of K, Q, V: (batch_size, n, d_model)
        # assert tf.shape(K) == tf.shape(Q) == tf.shape(V), "K, Q, V must have the same shape"
        batch_size, n, d_k = tf.shape(K).numpy()

        K = reshape_for_multihead_attention(K, batch_size, n, self.n_heads, self.d_k)  
        Q = reshape_for_multihead_attention(Q, batch_size, n, self.n_heads, self.d_k)
        V = reshape_for_multihead_attention(V, batch_size, n, self.n_heads, self.d_k)
        
        attention_output = self.linear_attention(K, Q, V)
        # Concat heads with transpose and reshape
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3]) # shape (batch_size, n, n_heads, d_k)
        attention_output = tf.reshape(attention_output, shape=[batch_size, n, self.d_model]) # shape (batch_size, n, d_model)
        outputs = self.w_o(attention_output)

        return outputs

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, k: int, d_model: int, d_ff: int, n_heads: int, dropout: float, full_attn: bool):
        super().__init__()
        self.mha = MultiHeadLinearAttention(k=k, d_model=d_model, n_heads=n_heads, dropout=dropout, full_attn=full_attn)
        self.ff = MLP(d_ff=d_ff, d_output=d_model, activation="gelu", dropout=dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = x + self.mha(x, x, x)
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x

class Encoder(tf.keras.Model):
    def __init__(self, n_layers: int, k: int, d_model: int, d_ff: int, n_heads: int, dropout: float, full_attn: bool):
        super().__init__()
        self.model = tf.keras.Sequential([EncoderLayer(k=k, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, full_attn=full_attn) for _ in range(n_layers)])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)

class Linformer(tf.keras.Model):
    def __init__(self, k: int, max_len: int, vocab_size: int, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float, full_attn: bool):
        super().__init__()
        self.embeddings_layer = EmbeddingsLayer(max_len=max_len, vocab_size=vocab_size, d_model=d_model)
        self.encoder = Encoder(n_layers=n_layers, k=k, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, full_attn=full_attn)
        self.pooler = GlobalAveragePooling1D()
        self.classifier = Dense(3, activation="softmax")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        embeddings = self.embeddings_layer(inputs)
        encoding = self.encoder(embeddings)
        pooled_output = self.pooler(encoding)
        outputs = self.classifier(pooled_output)
        return outputs

    def num_params(self) -> int:
        trainable_params = tf.math.reduce_sum(tf.size(v) for v in self.trainable_weights).numpy()
        non_trainable_params = tf.math.reduce_sum(tf.size(v) for v in self.non_trainable_weights).numpy()
        return trainable_params + non_trainable_params