import tensorflow as tf

import os
import numpy as np
import sys

from datasets import dataset_utils

from PIL import Image


def load_image(path):
    if type(path) is str:
        image_load = Image.open(path)
        image_load = image_load.resize((224, 224))
        image_load = image_load.convert('RGB')
        image_np = np.array(image_load.getdata()).reshape((224, 224, 3)).astype(np.uint8)
        return np.expand_dims(image_np, axis=0)
    elif type(path) is list:
        image_nps = []
        for one_path in path:
            image_load = Image.open(one_path)
            image_load = image_load.resize((224, 224))
            image_load = image_load.convert('RGB')
            image_np = np.array(image_load.getdata()).reshape((224, 224, 3)).astype(np.uint8)
            image_nps.append(image_np)
        return np.array(image_nps)


def search_images(dataset_dir):

    files_all = []
    labels = []
    class_num = 0
    for root, _, files in os.walk(dataset_dir):
        file_num = len(files)
        if file_num > 0 and root != dataset_dir:
            files_all.extend([os.path.join(root, file) for file in files])
            labels.extend([class_num] * file_num)
            class_num = class_num + 1

    file_all_num = len(files_all)
    print("Find %d classes" % class_num)
    print("Find %d images" % file_all_num)

    return files_all, labels, file_all_num, class_num


def oneshot_data_gen(features, labels, samples, possible_classes, shot):

    labels = np.array(labels)
    features = np.array(features)
    num_classes = labels.max() + 1

    for i in range(samples):
        sys.stdout.write('\r>> Encoding sample %d/%d' % (i, samples))
        sys.stdout.flush()

        oneshot_feature = []
        oneshot_label = []

        chosen_classes = np.random.choice(num_classes, possible_classes, replace=False)
        random_labels = np.random.choice(possible_classes, possible_classes, replace=False)

        for j in range(possible_classes):
            feature_index = np.where(labels == chosen_classes[j])[0]
            if j == 0:
                chosen_index = np.random.choice(feature_index, shot + 1, replace=False)
                chosen_label = [random_labels[j]] * (shot + 1)
            else:
                chosen_index = np.random.choice(feature_index, shot, replace=False)
                chosen_label = [random_labels[j]] * shot

            oneshot_feature.extend(features[chosen_index])
            oneshot_label.extend(chosen_label)

        oneshot_feature = np.array(oneshot_feature)
        oneshot_label = np.array(oneshot_label)

        rand_index = np.arange(1, shot * possible_classes + 1)
        np.random.shuffle(rand_index)

        oneshot_feature[1:] = oneshot_feature[rand_index]
        oneshot_label[1:] = oneshot_label[rand_index]

        yield oneshot_feature, oneshot_label


def oneshot_to_tfexample(feature_map, class_id):
    num = len(feature_map)  # samples num in one train item, including support set and target
    feature = {}

    for i in range(num):
        feature['feature%d' % i] = dataset_utils.bytes_feature(feature_map[i].tostring())
        feature['label%d' % i] = dataset_utils.int64_feature(class_id[i])

    return tf.train.Example(features=tf.train.Features(feature=feature))


def add_to_tfrecord(oneshot_data, tfrecord_writer, is_feature):

    if is_feature:
        for feature, label in oneshot_data:
            example = oneshot_to_tfexample(feature, label)
            tfrecord_writer.write(example.SerializeToString())
    else:
        for feature, label in oneshot_data:
            imgs = load_image(list(feature))
            example = oneshot_to_tfexample(imgs, label)
            tfrecord_writer.write(example.SerializeToString())


def encode_image(checkpoint_path, output_node, dataset_dir, batch_size, device):

    encode_graph = tf.Graph()
    with encode_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    inputs = encode_graph.get_tensor_by_name('input:0')
    logits = encode_graph.get_tensor_by_name('%s:0' % output_node)

    files_all, labels, file_all_num, class_num = search_images(dataset_dir)

    process_batch_num = int(file_all_num / batch_size)

    features = []
    with tf.device("/device:%s:0" % device):
        with tf.Session(graph=encode_graph) as sess:
            i = 0
            while i < process_batch_num:
                sys.stdout.write('\r>> Encoding image %d/%d' % (i * batch_size, file_all_num))
                sys.stdout.flush()
                images = load_image(files_all[i * batch_size: (i+1) * batch_size])
                logit = sess.run(logits, feed_dict={inputs: images})
                features.extend(logit)
                i = i + 1

            if i * batch_size < file_all_num - 1:
                sys.stdout.write('\r>> Encoding image %d/%d' % (i * batch_size, file_all_num))
                sys.stdout.flush()
                images = load_image(files_all[i * batch_size:])
                logit = sess.run(logits, feed_dict={inputs: images})
                features.extend(logit)

    features = np.squeeze(features)
    labels = np.squeeze(labels)

    return features, labels


def add_oneshot_to_tfrecord(features, labels, save_name, is_feature, possible_classes, shot, samples):

    oneshot_data = oneshot_data_gen(features, labels, samples=samples,
                                    possible_classes=possible_classes, shot=shot)

    with tf.python_io.TFRecordWriter(save_name) as tfrecord_writer:
        add_to_tfrecord(oneshot_data, tfrecord_writer, is_feature)

