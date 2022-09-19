import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation, GlobalAveragePooling1D
from utils import reshape_for_multihead_attention

from linformer import EmbeddingsLayer, MLP, LinearSelfAttention, MultiHeadLinearAttention, EncoderLayer, Encoder, Linformer

class EmbeddingsTest(tf.test.TestCase):
    def setUp(self):
        super(EmbeddingsTest, self).setUp()
        self.embeddings_layer = EmbeddingsLayer(max_len=2048, vocab_size=10000, d_model=128)

    def test_shape(self):
        x = tf.random.uniform((8, 2048), minval=0, maxval=9999)
        y = self.embeddings_layer(x)
        self.assertEqual(y.shape, (8, 2048, 128))


class MLPTest(tf.test.TestCase):
    def setUp(self):
        super(MLPTest, self).setUp()
        self.mlp = MLP(d_ff=512, d_output=128, activation='gelu', dropout=0.1)

    def test_shape(self):
        x = tf.random.uniform((8, 2048, 128))
        y = self.mlp(x)
        self.assertEqual(y.shape, (8, 2048, 128))


class LinearSelfAttentionTest(tf.test.TestCase):
    def setUp(self):
        super(LinearSelfAttentionTest, self).setUp()

        self.linear_self_attention = LinearSelfAttention(k=128, d_k=32, dropout=0.1, full_attn=False)

    def test_shape(self):
        x = tf.random.normal((8, 4, 1024, 32))
        y = self.linear_self_attention(x, x, x)
        self.assertEqual(y.shape, (8, 4, 1024, 32))
        

class MultiHeadLinearAttentionTest(tf.test.TestCase):
    def setUp(self):
        super(MultiHeadLinearAttentionTest, self).setUp()

        self.multi_head_linear_attention = MultiHeadLinearAttention(k=128, d_model=128, n_heads=4, dropout=0.1, full_attn=False)

    def test_shape(self):
        x = tf.random.normal((8, 1024, 128))
        y = self.multi_head_linear_attention(x, x, x)
        self.assertEqual(y.shape, (8, 1024, 128))

    def test_reshape_for_multihead_attention(self):
        x = tf.random.normal((8, 1024, 128))
        y = reshape_for_multihead_attention(x, 8, 1024, 4, 32)
        self.assertEqual(y.shape, (8, 4, 1024, 32))

class EncoderLayerTest(tf.test.TestCase):
    def setUp(self):
        super(EncoderLayerTest, self).setUp()
        self.encoder_layer = EncoderLayer(k=128, d_model=128, d_ff=512, n_heads=4, dropout=0.1, full_attn=False)

    def test_shape(self):
        x = tf.random.normal((8, 1024, 128))
        y = self.encoder_layer(x)
        self.assertEqual(y.shape, (8, 1024, 128))

class EncoderTest(tf.test.TestCase):
    def setUp(self):
        super(EncoderTest, self).setUp()

        self.encoder = Encoder(n_layers=2, k=128, d_model=128, d_ff=512, n_heads=4, dropout=0.1, full_attn=False)

    def test_shape(self):
        x = tf.random.normal((8, 1024, 128))
        y = self.encoder(x)
        self.assertEqual(y.shape, (8, 1024, 128))

class LinformerTest(tf.test.TestCase):
    def setUp(self):
        super(LinformerTest, self).setUp()

        self.linformer = Linformer(k=128, max_len=1024, vocab_size=10000, d_model=128, d_ff=512, n_heads=4, n_layers=2, dropout=0.1, full_attn=False)

    def test_shape(self):
        x = tf.random.uniform((8, 1024), minval=0, maxval=9999)
        y = self.linformer(x)
        self.assertEqual(y.shape, (8, 3))

if __name__ == "__main__":
    tf.test.main()