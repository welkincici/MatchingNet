from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

_ITEMS_TO_DESCRIPTIONS = {
    'feature': 'A [224 x 224 x 3] color image or a image encoded by several layers of CNN.',
    'label': 'A single integer between 0 and possible classes-1',
}


class Vector(slim.tfexample_decoder.ItemHandler):

    def __init__(self,
                 decode_key,
                 dtype=tf.float32):
        super(Vector, self).__init__([decode_key])
        self._decode_key = decode_key
        self._dtype = dtype

    def tensors_to_item(self, keys_to_tensors):
        image_buffer = keys_to_tensors[self._decode_key]

        return tf.decode_raw(image_buffer, self._dtype)


def get_split(data_source, samples, num_samples, vector_size, reader=None):
    if not reader:
        reader = tf.TFRecordReader

    keys_to_features = {}
    items_to_handlers = {}

    for i in range(samples):
        keys_to_features['feature%d' % i] = tf.FixedLenFeature((), tf.string, default_value='')
        keys_to_features['label%d' % i] = \
            tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        items_to_handlers['feature%d' % i] = Vector(decode_key='feature%d' % i)
        items_to_handlers['label%d' % i] = slim.tfexample_decoder.Tensor('label%d' % i)

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=data_source,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=vector_size,
        labels_to_names=None)

