import tensorflow as tf

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation

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
    def __init__(self, k: int, dropout: float):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.e = Dense(k)
        self.f = Dense(k)

    def call(self, K: tf.Tensor, Q: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
        # shape of K, Q, V: (batch_size, seq_len, d_k)
        K = self.e(K)
        K = tf.transpose(K, perm=[0, 2, 1])
        QK = tf.matmul(Q, K) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        P_bar = tf.nn.softmax(QK, axis=-1)
        P_bar = self.dropout(P_bar)
        V = self.f(V)
        return tf.matmul(P_bar, V)

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x


class Linformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense1(inputs)
        return self.dense2(x)

    def num_params(self) -> int:
        trainable_params = tf.math.reduce_sum(tf.size(v) for v in self.trainable_weights).numpy()
        non_trainable_params = tf.math.reduce_sum(tf.size(v) for v in self.non_trainable_weights).numpy()
        return trainable_params + non_trainable_params
