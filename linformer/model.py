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

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x


class Linformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, input: tf.Tensor) -> tf.Tensor:
        x = self.dense1(inputs)
        return self.dense2(x)