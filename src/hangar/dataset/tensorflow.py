from functools import partial
import tensorflow as tf


def yield_data(accessors, keys):
    for name in keys:
        yield tuple([acc.get_by_name(name) for acc in accessors])


def make_tf_dataset(accessors, keys):
    generator = partial(yield_data, accessors, keys)
    types = []
    shapes = []
    for acc in accessors:
        sample = acc.get_by_name(keys[0])

        # get shape
        if acc.schema_type == 'variable_shape':
            shape = (None,) * len(sample.shape)
        else:
            shape = sample.shape
        shapes.append(tf.TensorShape(shape))

        # get type
        types.append(tf.as_dtype(sample.dtype))
    shapes, types = tuple(shapes), tuple(types)
    return tf.data.Dataset.from_generator(generator=generator, output_types=types,
                                          output_shapes=shapes)
