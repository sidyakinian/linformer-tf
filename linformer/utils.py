import tensorflow as tf

def reshape_for_multihead_attention(M, batch_size, n, n_heads, d_k):
    M = tf.reshape(M, shape=[batch_size, n, n_heads, d_k])
    M = tf.transpose(M, perm=[0, 2, 1, 3]) # shape (batch_size, n_heads, n, d_k)
    return M