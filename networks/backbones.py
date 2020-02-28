import tensorflow as tf
from tensorflow.compat.v1.nn import max_pool2d

from .common import conv_bn_relu


def alexnet(input_t):
    output_t = input_t

    with tf.variable_scope('AlexNet') as scope:
        # conv
        output_t = conv_bn_relu(output_t, name='block_1', out_channel=96, strides=2, ksize=11)
        # pool, the original pytorch implementation use maxpooling with ceil_mode, which is not employ here
        output_t = max_pool2d(output_t, name='pool_1', ksize=3, strides=2, padding='VALID')
        # conv
        output_t = conv_bn_relu(output_t, name='block_2', out_channel=256, strides=1, ksize=5)
        # pool, the original pytorch implementation use maxpooling with ceil_mode, which is not employ here
        output_t = max_pool2d(output_t, name='pool_2', ksize=3, strides=2, padding='VALID')
        # conv
        output_t = conv_bn_relu(output_t, name='block_3', out_channel=384, strides=1, ksize=3)
        # conv
        output_t = conv_bn_relu(output_t, name='block_4', out_channel=384, strides=1, ksize=3)
        # conv
        output_t = conv_bn_relu(output_t, name='block_5', out_channel=256, strides=1, ksize=3, has_relu=False)

    return output_t