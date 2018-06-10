import os
import sys

import numpy as np
import tensorflow as tf
import cv2
import time

_VECTOR_SIZE = 80
_CLASSES = 10
_SHOT = 5
_PIC_SIZE = 224

encode_graph = tf.Graph()
with encode_graph.as_default():
    encode_graph_def = tf.GraphDef()
    with tf.gfile.GFile('model/frozen_oneshot_base2.pb', 'rb') as fid:
        encode_serialized_graph = fid.read()
        encode_graph_def.ParseFromString(encode_serialized_graph)
        tf.import_graph_def(encode_graph_def, name='')


match_graph = tf.Graph()
with match_graph.as_default():
    match_graph_def = tf.GraphDef()
    with tf.gfile.GFile('model/oneshot_nfce10_5.pb', 'rb') as fid:
        match_serialized_graph = fid.read()
        match_graph_def.ParseFromString(match_serialized_graph)
        tf.import_graph_def(match_graph_def, name='')

sess_encode = tf.Session(graph=encode_graph)
sess_match = tf.Session(graph=match_graph)

image = encode_graph.get_tensor_by_name('input:0')
logits = encode_graph.get_tensor_by_name('MobilenetV1/Logits/SpatialSqueeze:0')
base_prediction = encode_graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
features = match_graph.get_tensor_by_name('input:0')
labels = match_graph.get_tensor_by_name('support_labels:0')
match_prediction = match_graph.get_tensor_by_name('MatchNet/Squeeze_50:0')


def set_text(img, text, font, size, thick):
    letter_high = int(size * 34)
    for i, txt in enumerate(text.split('\n')):
        cv2.putText(img, txt, (0, letter_high * (i + 1)), font, size, (255, 255, 255), thick)


if os.path.exists("datasets/demo/support_set.npy") and os.path.exists("datasets/demo/labels.txt"):
    support_features = list(np.load("datasets/demo/support_set.npy"))
    class_name = list(np.loadtxt("datasets/demo/labels.txt", dtype=np.str))
else:
    support_features = list(np.random.random((_CLASSES * _SHOT, _VECTOR_SIZE)))
    class_name = ["NONE"] * _CLASSES

support_labels = np.reshape([[i]*_SHOT for i in range(10)], -1)

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
                    cv2.resize(frame, (_PIC_SIZE, _PIC_SIZE)), axis=0)})
            encoded_target = encoded_target[0]

            match_features = np.concatenate((np.array(encoded_target), np.array(support_features)))
            match_prdt = sess_match.run([match_prediction],
                                        feed_dict={features: match_features, labels: support_labels})
            match_prdt = np.squeeze(match_prdt[0])

            sort_index = np.argsort(match_prdt)
            msg = ""
            for i in [-1, -2, -3]:
                msg = msg + "%s:%f\n" % (class_name[sort_index[i]], match_prdt[sort_index[i]])

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
            msg = "Enter c to capture %d pictures" % _SHOT
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
                                                         cv2.resize(frame, (_PIC_SIZE, _PIC_SIZE)), axis=0)})
                    encoded_target = encoded_target[0][0]
                    support_set_buffer.append(encoded_target)

                    if count_buffer == _SHOT:
                        support_features.extend(support_set_buffer)
                        class_name.append(new_label_buffer)
                        support_features = support_features[-_SHOT * _CLASSES:]
                        class_name = class_name[-_CLASSES:]
                        np.save("datasets/demo/support_set.npy",
                                np.array(support_features))
                        np.savetxt("datasets/demo/labels.txt", class_name, fmt="%s")
                        flag = 0
                    else:
                        flag = 2

                    break

                elif boardkey == ord("n"):
                    flag = 2
                    break

        set_text(img, msg, font, 1.2, 2)
        cv2.imshow("demo", img)



