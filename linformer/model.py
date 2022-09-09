import tensorflow as tf

from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

class MLP(tf.keras.layers.Layer):
    def __init__(self, d_ff, d_output, activation, dropout):
        super().__init__()
        self.fc1 = Dense(d_ff, activation=activation)
        self.fc2 = Dense(d_output)
        self.dropout = Dropout(dropout)

    def call(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Linformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)