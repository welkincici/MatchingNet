from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def matchnet(inputs,
             support_labels,
             num_classes,
             fc_num=0,
             fce=True,
             batch_size=32,
             processing_steps=10,
             scope='MatchNet'):

    with tf.variable_scope(scope, 'MatchNet', [inputs, support_labels]):
        end_points = {}
        with tf.variable_scope('FC'):
            for i in range(fc_num):
                inputs = slim.fully_connected(inputs, num_classes,
                                              activation_fn=tf.nn.relu, scope='fc_%d' % i)
                end_points['fc_%d' % i] = inputs

        features = tf.unstack(inputs, axis=1)
        targets = features[0]
        support_features = features[1:]

        if fce:
            with tf.variable_scope('G_embedding'):
                fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_classes / 2)
                bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_classes / 2)
                # fw_cell = tf.contrib.rnn.DropoutWrapper(cell=fw_cell, input_keep_prob=1.0, output_keep_prob=0.8)
                # bw_cell = tf.contrib.rnn.DropoutWrapper(cell=bw_cell, input_keep_prob=1.0, output_keep_prob=0.8)

                outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                                                                      support_features,
                                                                                      dtype=tf.float32)

                g_embedding = tf.add(tf.stack(support_features), tf.stack(outputs))

            with tf.variable_scope('F_embedding'):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_classes)
                # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=0.8)

                state = cell.zero_state(batch_size, tf.float32)

                for step in range(processing_steps):
                    h_k_1 = tf.add(state[1], targets)
                    a_k_1 = [tf.matmul(tf.expand_dims(h_k_1, 1), tf.expand_dims(g_i, 2))
                             for g_i in tf.unstack(g_embedding)]
                    a_k_1 = tf.nn.softmax(tf.transpose(tf.stack(a_k_1), [1, 2, 3, 0]))
                    a_k_1 = tf.transpose(a_k_1, [3, 0, 1, 2])
                    r_k_1 = tf.reduce_sum(tf.matmul(tf.expand_dims(g_embedding, 3), a_k_1), axis=0)
                    state = tf.contrib.rnn.LSTMStateTuple(state[0], tf.add(h_k_1, tf.squeeze(r_k_1)))
                    f_embedding, state = cell(targets, state)  # state:[c, h]

                    tf.get_variable_scope().reuse_variables()

        else:
            g_embedding = support_features
            f_embedding = targets

        cos_sim_list = []
        for i in tf.unstack(g_embedding):
            target_normed = f_embedding
            i_normed = tf.nn.l2_normalize(i, 1)
            similarity = tf.matmul(tf.expand_dims(target_normed, 1), tf.expand_dims(i_normed, 2))
            cos_sim_list.append(tf.squeeze(similarity, [1, ]))

        cos_sim = tf.concat(axis=1, values=cos_sim_list)
        weighting = tf.nn.softmax(cos_sim)
        label_prob = tf.squeeze(tf.matmul(tf.expand_dims(weighting, 1), support_labels), axis=1, name='pred')

        end_points['G_embedding'] = g_embedding
        end_points['F_embedding'] = f_embedding
        end_points['Cos_sim'] = cos_sim
        end_points['Weighting'] = weighting
        end_points['Label_prob'] = label_prob

        local_var = tf.global_variables()
        for var in local_var:
            slim.add_model_variable(var)
            tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)

        return label_prob, end_points
