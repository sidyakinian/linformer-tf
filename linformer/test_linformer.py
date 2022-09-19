import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Activation, GlobalAveragePooling1D
from utils import reshape_for_multihead_attention

from linformer import EmbeddingsLayer, MLP, LinearSelfAttention, MultiHeadLinearAttention, EncoderLayer, Encoder, Linformer

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

if __name__ == "__main__":
    tf.test.main()