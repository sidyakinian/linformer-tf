import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from linformer.model import Linformer

## Data preprocessing

(ds_train, ds_test), ds_info = tfds.load(
    'multi_nli',
    split=['train[:2%]', 'validation_matched[:2%]'],
    shuffle_files=True,
    with_info=True,
)

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(len(ds_train))
ds_train = ds_train.batch(64, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.batch(64, drop_remainder=True)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

vectorization_layer = TextVectorization(
    max_tokens=10_000,
    output_mode='int',
    output_sequence_length=2048)

def to_text(element):
    prefixes = tf.repeat("premise: ", repeats=64)
    premises = element["premise"]
    separators = tf.repeat(". hypothesis: ", repeats=64)
    hypotheses = element["hypothesis"]

    assert prefixes.shape == premises.shape == hypotheses.shape, f"shapes must batch, but got {prefixes.shape}, {premises.shape}, {hypotheses.shape}"

    text_element = tf.strings.join([prefixes, premises, separators, hypotheses])
    return text_element

train_text = ds_train.map(to_text)
vectorization_layer.adapt(train_text)

## Convert dataset to training format

def to_traintest_data(element):
    prefixes = tf.repeat("premise: ", repeats=64)
    premises = element["premise"]
    separators = tf.repeat(". hypothesis: ", repeats=64)
    hypotheses = element["hypothesis"]

    text_element = tf.strings.join([prefixes, premises, separators, hypotheses])
    vectorized_element = vectorization_layer(text_element)
    label_element = element["label"]
    return (vectorized_element, label_element)

training_dataset = ds_train.map(to_traintest_data)
testing_dataset = ds_test.map(to_traintest_data)

## Fit model

model = Linformer(k=64, max_len=2048, vocab_size=10_000, d_model=512, d_ff=2048, n_heads=8, n_layers=3, dropout=0.1, full_attn=False)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(training_dataset, epochs=3, validation_data=testing_dataset)