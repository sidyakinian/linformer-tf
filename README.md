# Modified Linformer implementation in Tensorflow 2.0

## Description

This is a modified implementation of the [Linformer paper](https://arxiv.org/abs/2006.04768) (Wang et. al., 2020). A major bottleneck in transformer performance is the time and space complexity of $O(n^2)$ during the self-attention operation, where $n$ is the length of the input sequence, since every word in the sequence needs to attend to every other word for a total of $\Theta(n^2)$ connections. This paper aims to reduce both time and space complexity of this bottleneck by projecting key and value matrices from dimension $n$ to dimension $k$. This works in theory because the key and value matrices are of shape $n \times d_{model}$ and thus can only have a maximum rank of $d_{model}$.

This version is modified for sequence classification tasks (e. g. natural language inference, sentiment analysis, etc), and is thus encoder-only. To compensate for the reduced complexity of the model, one can increase the number of layers in the encoder (this would have the same effect as the decoder would receive no previous output tokens, and the cross-attention layer would effectively be skipped).

## Usage

You can initialize, compile, and fit the model as you would any other Keras model. If you're unfamiliar with Keras, please see [documentation](https://keras.io/api/models/). For example:

```
model = Linformer(k=256, max_len=MAX_LEN, vocab_size=20_000, d_model=256, d_ff=1024, n_heads=4, n_layers=3, dropout=0.1, full_attn=False, batch_size=16)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"], run_eagerly=True)
model.fit(x=x_train, y=y_train, batch_size=16, epochs=25, validation_data = (x_val, y_val), callbacks=[tensorboard_callback])
```

## Known issues

* Batch size is passed on model initialization, because otherwise it's difficult to get the batch size from tensor shape unless the code is run eagerly (but eager execution is slow compared to graph-optimized execution).
* Number of parameters function isn't tested.
