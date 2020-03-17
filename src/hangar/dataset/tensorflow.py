from functools import partial
import random

import tensorflow as tf
from .common import Dataset


def yield_data(dataset, shuffle):
    if shuffle:
        random.shuffle(dataset.keys)
    for name in dataset.keys:
        yield dataset[name]


def make_tensorflow_dataset(columns, keys=None, shuffle=False):
    dataset = Dataset(columns, keys)
    generator = partial(yield_data, dataset, shuffle)
    types = []
    shapes = []
    for col in dataset.columns:

        # get shape
        if col.schema_type == 'variable_shape':
            shape = (None,) * len(col.shape)
        else:
            shape = col.shape
        shapes.append(tf.TensorShape(shape))

        # get type
        types.append(tf.as_dtype(col.dtype))
    shapes, types = tuple(shapes), tuple(types)
    return tf.data.Dataset.from_generator(generator=generator, output_types=types,
                                          output_shapes=shapes)
