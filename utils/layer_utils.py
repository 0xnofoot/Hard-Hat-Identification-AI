# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 设置阈值函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 设置卷积层
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 设置池化层
# def pool(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
def GAU_block(low_feature, high_feature, channel_out):
    # channel_out = outshape[3]
    high_feature = tf.reduce_mean(high_feature, [1, 2], keep_dims=True)
    # high_feature = conv2d(high_feature,channel_out, 1)
    # low_feature = conv2d(low_feature,channel_out,3)
    high_feature = slim.conv2d(high_feature, channel_out, 1, stride=1, activation_fn=tf.nn.relu,
                               padding='SAME', weights_initializer=slim.initializers.xavier_initializer())
    low_feature = slim.conv2d(low_feature, channel_out, 3, stride=1,
                              padding='SAME', weights_initializer=slim.initializers.xavier_initializer())
    out = high_feature * low_feature
    return out


def SE_block(x, ratio):
    shape = x.get_shape().as_list()
    w = shape[1]
    h = shape[2]
    channel_out = shape[3]
    # print(shape)
    with tf.variable_scope("squeeze_and_excitation"):
        # 第一层，全局平均池化层
        # squeeze = tf.nn.avg_pool(x, [1, shape[1], shape[2], 1], [1, shape[1], shape[2], 1], padding="SAME")
        squeeze = tf.reduce_mean(x, [1, 2], keep_dims=True)

        # 第二层，全连接层
        w_excitation1 = weight_variable([1, 1, channel_out, channel_out // ratio])
        b_excitation1 = bias_variable([channel_out / ratio])
        excitation1 = tf.nn.conv2d(squeeze, w_excitation1, strides=[1, 1, 1, 1], padding="SAME") + b_excitation1
        excitation1_output = tf.nn.relu(excitation1)
        # 第三层，全连接层
        w_excitation2 = weight_variable([1, 1, channel_out // ratio, channel_out])
        b_excitation2 = bias_variable([channel_out])
        excitation2 = tf.nn.conv2d(excitation1_output, w_excitation2, strides=[1, 1, 1, 1],
                                   padding="SAME") + b_excitation2
        excitation2_output = tf.nn.sigmoid(excitation2)
        # 第四层，点乘
        excitation_output = tf.reshape(excitation2_output, [-1, 1, 1, channel_out])
        h_output = excitation_output * x

    return h_output


def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)
#        net = SE_block(net, 16)
        net = net + shortcut

        return net

    # first two conv2d layers
    net = conv2d(inputs, 32, 3, strides=1)
    net = conv2d(net, 64, 3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3


def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs

def PAM_Module(inputs):  # 空间注意力b*h*w*c
    shape1 = inputs.get_shape().as_list()
    gama = tf.Variable(tf.zeros(1), dtype=tf.float32)
    channal = shape1[3]
    shape2 = tf.shape(inputs)
    batch, h, w = shape2[0], shape2[1], shape2[2]
    # inputs=conv2d(inputs,channal,3)#加入卷积层
    proj_query = conv2d(inputs, channal // 8, 1)  # b*h*w*c/8
    proj_query = tf.reshape(proj_query, (batch, h * w, -1))  # b*n*c/8

    proj_key = conv2d(inputs, channal // 8, 1)
    proj_key = tf.reshape(proj_key, (batch, h * w, -1))  # b*n*c/8
    proj_key = tf.transpose(proj_key, perm=[0, 2, 1])  # b*c/8*n
    energy = tf.matmul(proj_query, proj_key)  # b*n*n
    attention = tf.nn.softmax(energy, axis=2)  # b*n*n
    attention = tf.transpose(attention, perm=[0, 2, 1])
    proj_value = conv2d(inputs, channal, 1)
    proj_value = tf.reshape(proj_value, (batch, h * w, -1))  # b*n*c
    proj_value = tf.transpose(proj_value, perm=[0, 2, 1])  ##b*c*n
    out = tf.matmul(proj_value, attention)
    out = tf.reshape(out, (batch, h, w, channal))
    out = gama * out + inputs
    out = conv2d(out, channal, 1)
    out = conv2d(out, channal, 3)
    return out


def CAM_Module(inputs):
    shape1 = inputs.get_shape().as_list()
    gama = tf.Variable(tf.zeros(1), dtype=tf.float32)
    channal = shape1[3]
    shape2 = tf.shape(inputs)
    batch, h, w = shape2[0], shape2[1], shape2[2]
    inputs = conv2d(inputs, channal, 3)
    proj_query = tf.reshape(inputs, (batch, -1, channal))  # b*n*c
    proj_query = tf.transpose(proj_query, perm=[0, 2, 1])  # b*c*n
    proj_key = tf.reshape(inputs, (batch, -1, channal))  # b*n*c
    enerty = tf.matmul(proj_query, proj_key)  # b*c*c
    enerty_new = tf.reduce_max(enerty, axis=-1, keep_dims=True)[0] - enerty
    attention = tf.nn.softmax(enerty_new, axis=2)
    proj_value = tf.reshape(inputs, (batch, -1, channal))  # b*n*c
    out = tf.matmul(proj_value, attention)
    out = tf.reshape(out, (batch, h, w, channal))
    out = gama * out + inputs
    out = conv2d(out, channal, 1)
    out = conv2d(out, channal, 3)
    return out