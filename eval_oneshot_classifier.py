# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import mini_imagenet_oneshot
import math
from nets import matchnet

import os

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are written to.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_string('data_source', None, 'The path of data source.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to eval.')

tf.app.flags.DEFINE_integer('processing_steps', 5, 'The number of process step.')

tf.app.flags.DEFINE_boolean('fce', True, 'Weather to use fully embedding')

tf.app.flags.DEFINE_integer('vector_size', None, 'The shape of input feature shape.')

tf.app.flags.DEFINE_integer('fc_num', 0, 'The number of fully-connected layers in front of '
                                         'match layers.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.data_source):
        raise ValueError("no such data source")

    msg = FLAGS.data_source.split('/')[-1]
    msg = msg.split('.')[0]
    msg = msg.split('_')
    possible_classes = int(msg[2])
    shot = int(msg[3])
    num_samples = int(msg[4])
    samples = possible_classes * shot + 1

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = mini_imagenet_oneshot.get_split(FLAGS.data_source, samples,
                                                  num_samples, FLAGS.vector_size)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        get_list = []
        for i in range(samples):
            get_list.append('feature%d' % i)
        for i in range(samples):
            get_list.append('label%d' % i)

        receive_list = provider.get(get_list)

        feature = tf.stack(receive_list[0:samples], axis=0)
        feature = tf.reshape(feature, [samples, dataset.num_classes])
        label = receive_list[samples]
        slabel = tf.stack(receive_list[samples + 1:], axis=0)
        slabel = slim.one_hot_encoding(slabel, possible_classes)

        features, labels, slabels = tf.train.batch(
            [feature, label, slabel],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        logits, _ = matchnet.matchnet(features, slabels, dataset.num_classes, FLAGS.fc_num,
                                      batch_size=FLAGS.batch_size,
                                      processing_steps=FLAGS.processing_steps, fce=FLAGS.fce)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore)


if __name__ == '__main__':
    tf.app.run()
