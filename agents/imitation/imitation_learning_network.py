from __future__ import print_function

import numpy as np

import tensorflow as tf


def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


# DROPOUT_VEC_TRAIN = [1.0] * 8 + [0.7] * 4 + [0.5] * 4 + [0.5] * 4 + [0.5, 1.] * 9
# DROPOUT_VEC_TRAIN = [1.0] * 11 + [0.7] * 4 + [0.5] * 4 + [0.5] * 4 + [0.5, 1.] * 9 # multi
DROPOUT_VEC_TRAIN = [1.0] * 19 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
DROPOUT_VEC_INFER = [1.0 for _ in DROPOUT_VEC_TRAIN]

class Network(object):

    def __init__(self, dropout, image_shape):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._conv_rate = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def atrous_conv(self, x, kernel_size, rate, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_{}'.format(self._count_conv))
        bias = bias_variable([output_size], name='B_c_{}'.format(self._count_conv))

        self._weights['W_conv{}'.format(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_rate.append(rate)
        conv_res = tf.add(tf.nn.atrous_conv2d(x, weights, rate, padding=padding_in,
                                       name='conv2d_{}'.format(self._count_conv)), bias,
                          name='add_{}'.format(self._count_conv))

        self._features['conv_block{}'.format(self._count_conv - 1)] = conv_res

        return conv_res

    def residual(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_{}'.format(self._count_conv))
        bias = bias_variable([output_size], name='B_c_{}'.format(self._count_conv))

        self._weights['W_conv{}'.format(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_{}'.format(self._count_conv)), bias,
                          name='add_{}'.format(self._count_conv))

        self._features['conv_block{}'.format(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def atrous_conv_block(self, x, kernel_size, rate, output_size, padding_in='SAME'):
        with tf.name_scope("conv_block{}".format(self._count_conv)):
            x = self.atrous_conv(x, kernel_size, rate, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)
            return self.activation(x)

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)

            return self.activation(x)

    def residual_block(self, x, kernel_size, stride, output_size, padding_in='SAME', change_dim=False):
        with tf.name_scope("residual_block{}".format(self._count_conv)):
            if change_dim:
                down_sample = self.max_pool(x)
                shortcut = self.conv_block(down_sample, 1, 1, output_size, padding_in='SAME')
                bn_1 = self.bn(x)
                relu_1 = self.activation(bn_1)
                conv_1 = self.residual(relu_1, kernel_size, 2, output_size,
                                       padding_in=padding_in) ### stride 2 DownSampling

                bn_2 = self.bn(conv_1)
                relu_2 = self.activation(bn_2)
                conv_2 = self.residual(relu_2, kernel_size, stride, output_size, padding_in=padding_in)

                x_output = conv_2 + shortcut
            else:
                shortcut = x
                bn_1 = self.bn(x)
                relu_1 = self.activation(bn_1)
                conv_1 = self.residual(relu_1, kernel_size, stride, output_size,
                                       padding_in=padding_in)

                bn_2 = self.bn(conv_1)
                relu_2 = self.activation(bn_2)
                conv_2 = self.residual(relu_2, kernel_size, stride, output_size, padding_in=padding_in)

                x_output = conv_2 + shortcut

            return x_output

    def fc_block(self, x, output_size):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features



def load_imitation_learning_network(input_image, input_data, input_size, dropout):
    branches = []
    dropout = DROPOUT_VEC_INFER
    x = input_image
    network_manager = Network(dropout, tf.shape(x))
    # xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
    #
    # """conv2"""
    # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    #
    # """conv3"""
    # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    #
    # """conv4"""
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')

    # xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='SAME')
    #
    # xc5 = network_manager.conv_block(x, 5, 2, 32, padding_in='SAME')
    #
    # xc7 = network_manager.conv_block(x, 7, 2, 32, padding_in='SAME')
    #
    # xc = tf.concat([xc, xc5, xc7], -1)
    #
    # """conv2"""
    # xc5 = network_manager.conv_block(xc, 5, 2, 64, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='SAME')
    #
    # xc = tf.concat([xc, xc5], -1)
    #
    # """conv3"""
    # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='SAME')
    #
    # """conv4"""
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
    # resnet
    xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
    res1 = network_manager.residual_block(xc, 3, 1, 32, padding_in='SAME', change_dim=False)

    res2 = network_manager.residual_block(res1, 3, 1, 32, padding_in='SAME', change_dim=False)
    res3 = network_manager.residual_block(res2, 3, 1, 32, padding_in='SAME', change_dim=False)

    # 22 * 100
    res4 = network_manager.residual_block(res3, 3, 1, 64, padding_in='SAME', change_dim=True)
    res5 = network_manager.residual_block(res4, 3, 1, 64, padding_in='SAME', change_dim=False)

    # 11 * 100
    res6 = network_manager.residual_block(res5, 3, 1, 128, padding_in='SAME', change_dim=True)
    res7 = network_manager.residual_block(res6, 3, 1, 128, padding_in='SAME', change_dim=False)

    res7 = network_manager.conv_block(res7, 1, 1, 256, padding_in='SAME')

    res8 = network_manager.residual_block(res7, 3, 1, 256, padding_in='SAME', change_dim=False)
    res9 = network_manager.residual_block(res8, 3, 1, 256, padding_in='SAME', change_dim=False)
    ####################################################
    # xx = network_manager.conv_block(x, 3, 1, 32, padding_in='SAME')
    #
    # xc = network_manager.conv_block(xx, 3, 1, 32, padding_in='SAME')
    # xc5 = network_manager.atrous_conv_block(xx, 3, 1, 32, padding_in='SAME')
    # xc7 = network_manager.atrous_conv_block(xx, 3, 2, 32, padding_in='SAME')
    # xc11 = network_manager.atrous_conv_block(xx, 3, 3, 32, padding_in='SAME')
    #
    # concat = tf.concat([xc, xc5, xc7, xc11], -1)
    # # xx = network_manager.conv_block(xx, 3, 2, 64, padding_in='SAME')
    # xx = network_manager.max_pool(concat)
    #
    # """conv2"""
    # xc = network_manager.conv_block(xx, 3, 1, 64, padding_in='SAME')
    # xc5 = network_manager.atrous_conv_block(xx, 3, 1, 64, padding_in='SAME')
    # xc7 = network_manager.atrous_conv_block(xx, 3, 2, 64, padding_in='SAME')
    # xc11 = network_manager.atrous_conv_block(xx, 3, 3, 64, padding_in='SAME')
    #
    # concat = tf.concat([xc, xc5, xc7, xc11], -1)
    # # xx = network_manager.conv_block(xx, 3, 2, 128, padding_in='SAME')
    # xx = network_manager.max_pool(concat)
    #
    # """conv3"""
    # xc = network_manager.conv_block(xx, 3, 1, 128, padding_in='SAME')
    # xc5 = network_manager.atrous_conv_block(xx, 3, 1, 128, padding_in='SAME')
    # xc7 = network_manager.atrous_conv_block(xx, 3, 2, 128, padding_in='SAME')
    # xc11 = network_manager.atrous_conv_block(xx, 3, 3, 128, padding_in='SAME')
    #
    # concat = tf.concat([xc, xc5, xc7, xc11], -1)
    # # xc = network_manager.conv_block(xx, 3, 1, 256, padding_in='SAME')
    # xc = network_manager.max_pool(concat)
    # """conv1 - 44"""
    # xc = network_manager.conv_block(x, 3, 2, 32, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='SAME')
    #
    # xc5 = network_manager.conv_block(x, 5, 2, 32, padding_in='SAME')
    #
    # xc7 = network_manager.conv_block(x, 7, 2, 32, padding_in='SAME')
    #
    # xc = tf.concat([xc, xc5, xc7], -1)
    #
    # """conv2"""
    # xc5 = network_manager.conv_block(xc, 5, 2, 64, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='SAME')
    #
    # xc = tf.concat([xc, xc5], -1)
    #
    # """conv3"""
    # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='SAME')
    #
    # """conv4"""
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
    """conv3"""
    # xc5 = network_manager.conv_block(xc, 5, 2, 64, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='SAME')
    #
    # xc = tf.concat([xc, xc5], -1)
    #
    # """conv3"""
    # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='SAME')
    #
    # """conv4"""
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='SAME')



    # """conv1"""  # kernel sz, stride, num feature maps
    # xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
    #
    # """conv2"""
    # xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    #
    # """conv3"""
    # xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    #
    # """conv4"""
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    # xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    """mp3 (default values)"""
    # multi_scale = tf.concat([xc, xc1], 1)
    """ reshape """
    x = tf.reshape(res9, [-1, int(np.prod(res9.get_shape()[1:]))], name='reshape')
    print(x)

    """ fc1 """
    x = network_manager.fc_block(x, 512)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512)

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data[1]  # get the speed from input data
        speed = network_manager.fc_block(speed, 128)
        speed = network_manager.fc_block(speed, 128)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512)

    """Start BRANCHING"""
    branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], \
                     ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]]

    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256)
                branch_output = network_manager.fc_block(branch_output, 256)
            else:
                branch_output = network_manager.fc_block(j, 256)
                branch_output = network_manager.fc_block(branch_output, 256)

            branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        print(branch_output)

    return branches
