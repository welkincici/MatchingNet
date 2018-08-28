# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from nets import matchnet


slim = tf.contrib.slim

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_string('output_file', None, 'Where to save the resulting file to.')

tf.app.flags.DEFINE_integer('possible_classes', None, 'Number of possible classes.')

tf.app.flags.DEFINE_integer('shot', None, 'Number of support samples in each possible class')

tf.app.flags.DEFINE_integer('fc_num', 0, 'The number of fully-connected layers in front of '
                                         'match layers.')

tf.app.flags.DEFINE_integer('vector_size', None, 'The shape of input feature shape.')

tf.app.flags.DEFINE_integer('processing_steps', 5, 'The number of process step.')

tf.app.flags.DEFINE_boolean('fce', True, 'Weather to use fully embedding')

tf.app.flags.DEFINE_boolean('show_nodes', True, 'Weather to show graph nodes')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
        features = tf.placeholder(name='input', dtype=tf.float32,
                                  shape=[FLAGS.possible_classes * FLAGS.shot + 1, FLAGS.vector_size])
        support_labels = tf.placeholder(name='support_labels', dtype=tf.int32,
                                        shape=[FLAGS.possible_classes * FLAGS.shot, ])

        support_labels = slim.one_hot_encoding(support_labels, FLAGS.possible_classes)
        features = tf.expand_dims(features, axis=0)
        support_labels = tf.expand_dims(support_labels, axis=0)

        logits, _ = matchnet.matchnet(features, support_labels, FLAGS.vector_size, FLAGS.fc_num,
                                      batch_size=1,
                                      processing_steps=FLAGS.processing_steps,
                                      fce=FLAGS.fce)

        graph_def = graph.as_graph_def()
        with gfile.GFile(FLAGS.output_file, 'wb') as f:
            f.write(graph_def.SerializeToString())

        if FLAGS.show_nodes:
            print(graph_def)


if __name__ == '__main__':
    tf.app.run()
