import os

import numpy as np
import tensorflow as tf
import cv2
import time

tf.app.flags.DEFINE_integer('vector_size', 80, 'The size of encode graph output vector')

tf.app.flags.DEFINE_integer('num_classes', 10, 'Number of classes')

tf.app.flags.DEFINE_integer('shot', 5, 'Number of pictures in every support classes')

tf.app.flags.DEFINE_integer('pic_size', 224, 'Resized picture size')

tf.app.flags.DEFINE_string('encode_graph', 'model/frozen_oneshot_base.pb', 'Graph used to encode raw pictures '
                                                                           'in to representative vectors')

tf.app.flags.DEFINE_string('match_graph', 'model/oneshot_nfce10_5.pb', 'Graph used to match target '
                                                                       'vectors with support classes')

tf.app.flags.DEFINE_string('support_dir', 'support_data', 'Path of support data')

FLAGS = tf.app.flags.FLAGS

encode_graph = tf.Graph()
with encode_graph.as_default():
    encode_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.encode_graph, 'rb') as fid:
        encode_serialized_graph = fid.read()
        encode_graph_def.ParseFromString(encode_serialized_graph)
        tf.import_graph_def(encode_graph_def, name='')


match_graph = tf.Graph()
with match_graph.as_default():
    match_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FLAGS.match_graph, 'rb') as fid:
        match_serialized_graph = fid.read()
        match_graph_def.ParseFromString(match_serialized_graph)
        tf.import_graph_def(match_graph_def, name='')

sess_encode = tf.Session(graph=encode_graph)
sess_match = tf.Session(graph=match_graph)

support_set_path = "%s/support_set.npy" % FLAGS.support_dir
label_path = "%s/labels.txt" % FLAGS.support_dir
if os.path.exists(support_set_path) and os.path.exists(label_path):
    support_features = list(np.load(support_set_path))
    class_name = list(np.loadtxt(label_path, dtype=np.str))
else:
    support_features = list(np.random.random((FLAGS.num_classes * FLAGS.shot, FLAGS.vector_size)))
    class_name = ["NONE"] * FLAGS.num_classes

support_labels = np.reshape([[i]*FLAGS.shot for i in range(10)], -1)

image = encode_graph.get_tensor_by_name('input:0')
logits = encode_graph.get_tensor_by_name('MobilenetV1/Logits/SpatialSqueeze:0')
base_prediction = encode_graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
features = match_graph.get_tensor_by_name('input:0')
labels = match_graph.get_tensor_by_name('support_labels:0')
match_prediction = match_graph.get_tensor_by_name('MatchNet/pred:0')


def set_text(img, text, font, size, thick):
    letter_high = int(size * 34)
    for i, txt in enumerate(text.split('\n')):
        cv2.putText(img, txt, (0, letter_high * (i + 1)), font, size, (255, 255, 255), thick)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
time.sleep(2)
font = cv2.FONT_HERSHEY_SIMPLEX
msg = ""
new_label_buffer = ""
count_buffer = 0
support_set_buffer = []
flag = 0


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        img = np.zeros((640, 640, 3), np.uint8)
        img[160:, :, :] = img[160:, :, :] + frame

        boardkey = cv2.waitKey(1)

        if not flag:
            if boardkey == ord("c"):
                flag = 1
                new_label_buffer = ""
            if boardkey == ord("q"):
                break

            encoded_target = sess_encode.run(
                [logits], feed_dict={image: np.expand_dims(
                    cv2.resize(frame, (FLAGS.pic_size, FLAGS.pic_size)), axis=0)})
            encoded_target = encoded_target[0]

            match_features = np.concatenate((np.array(encoded_target), np.array(support_features)))
            match_prdt = sess_match.run([match_prediction],
                                        feed_dict={features: match_features, labels: support_labels})
            match_prdt = np.squeeze(match_prdt[0])

            sort_index = np.argsort(match_prdt)
            msg = ""
            for i in [-1, -2, -3]:
                msg = msg + "%s:%f\n" % (class_name[sort_index[i]], match_prdt[sort_index[i]])
            msg = msg + "Press 'c' to set a new class, 'q' to quit\n"
        elif flag == 1:
            if boardkey == -1:
                pass
            elif boardkey == ord("#"):
                flag = 0
                new_label_buffer = ""
            elif boardkey == ord("$"):
                new_label_buffer = ""
            elif boardkey == ord("&"):
                count_buffer = 0
                support_set_buffer = []
                flag = 2
            elif chr(boardkey).isalnum():
                new_label_buffer = new_label_buffer + chr(boardkey)

            msg = "Please enter a new label\n(#quit,$backspace,&submit):\n" + new_label_buffer
        elif flag == 2:
            if boardkey == ord("c"):
                flag = 3
            msg = "Enter c to capture %d pictures" % FLAGS.shot
        elif flag == 3:
            msg = "Is this picture clear?(y/n)"
            set_text(img, msg, font, 1.2, 2)
            cv2.imshow("demo", img)
            while 1:
                boardkey = cv2.waitKey(1)
                if boardkey == ord("y"):
                    count_buffer = count_buffer + 1
                    encoded_target = sess_encode.run([logits],
                                                     feed_dict={image: np.expand_dims(
                                                         cv2.resize(frame, (FLAGS.pic_size, FLAGS.pic_size)), axis=0)})
                    encoded_target = encoded_target[0][0]
                    support_set_buffer.append(encoded_target)

                    if count_buffer == FLAGS.shot:
                        support_features.extend(support_set_buffer)
                        class_name.append(new_label_buffer)
                        support_features = support_features[-FLAGS.shot * FLAGS.num_classes:]
                        class_name = class_name[-FLAGS.num_classes:]
                        np.save(support_set_path,
                                np.array(support_features))
                        np.savetxt(label_path, class_name, fmt="%s")
                        flag = 0
                    else:
                        flag = 2

                    break

                elif boardkey == ord("n"):
                    flag = 2
                    break

        set_text(img, msg, font, 1, 2)
        cv2.imshow("demo", img)


if __name__ == '__main__':
    tf.app.run()