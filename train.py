import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'multi_nli',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)