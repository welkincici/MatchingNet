import tensorflow as tf
import numpy as np
import os

from datasets import data_utils


slim = tf.contrib.slim

tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('possible_classes', None, 'Number of possible classes')

tf.app.flags.DEFINE_integer('shot', None, 'Number of samples in each support class.')

tf.app.flags.DEFINE_integer('samples', 10000, 'Number train sample to generate.')

# if 'dataset_dir' is a directory

tf.app.flags.DEFINE_boolean('from_raw_images', False, 'Weather to use raw pixels to train matching net.')

tf.app.flags.DEFINE_string('checkpoint_path', None, 'The frozen graph used to encode raw images.')

tf.app.flags.DEFINE_string('output_node', None, 'The output node.')

tf.app.flags.DEFINE_integer('batch_size', 50, 'Number of images encoded at once.')

tf.app.flags.DEFINE_string('device', 'GPU', 'The type of device used to encode image.')

tf.app.flags.DEFINE_boolean('save_encoded_images', True, 'Weather to save encoded images.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    if not FLAGS.possible_classes:
        raise ValueError('You must supply the number of possible classes with --possible_classes')
    if not FLAGS.shot:
        raise ValueError('You must supply the number of samples in each support class with --shot')

    is_feature = True
    if os.path.isdir(FLAGS.dataset_dir):
        if FLAGS.from_raw_images:
            features, labels, _, _ = data_utils.search_images(FLAGS.dataset_dir)
            is_feature = False
            save_name = "%s/oneshot_raw_%d_%d_%d.tfrecord" % (FLAGS.dataset_dir, FLAGS.possible_classes,
                                                              FLAGS.shot, FLAGS.samples)
        else:
            if not FLAGS.checkpoint_path:
                raise ValueError('You must supply the frozen graph used to encode '
                                 'raw images with --checkpoint_path')
            if not FLAGS.output_node:
                raise ValueError('You must supply the output node with --output_node')

            features, labels = data_utils.encode_image(FLAGS.checkpoint_path, FLAGS.output_node,
                                                       FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.device)
            node_abbr = FLAGS.output_node.split('/')[-1]

            if FLAGS.save_encoded_images:
                output_name = '%s/encoded_images_%s.npz' % (FLAGS.dataset_dir, node_abbr)
                np.savez(output_name, features=features, labels=labels)

            save_name = "%s/oneshot_%s_%d_%d_%d.tfrecord" % (FLAGS.dataset_dir, node_abbr,
                                                             FLAGS.possible_classes, FLAGS.shot,
                                                             FLAGS.samples)
    else:
        encoded_images = np.load(FLAGS.dataset_dir)
        save_info = FLAGS.dataset_dir.split('/')
        dataset_dir = '/'.join(save_info[:-1])
        node_abbr = save_info[-1].split('.')[0]
        node_abbr = node_abbr.split('_')[-1]
        features = encoded_images["features"]
        labels = encoded_images["labels"]
        save_name = "%s/oneshot_%s_%d_%d_%d.tfrecord" % (dataset_dir, node_abbr,
                                                         FLAGS.possible_classes, FLAGS.shot,
                                                         FLAGS.samples)

    data_utils.add_oneshot_to_tfrecord(features, labels,
                                       save_name, is_feature,
                                       FLAGS.possible_classes, FLAGS.shot, FLAGS.samples)


if __name__ == '__main__':
    tf.app.run()
