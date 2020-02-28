from tensorflow.compat.v1.nn import conv2d, bias_add, batch_normalization, relu
from tensorflow.compat.v1.nn import depthwise_conv2d
from tensorflow.compat.v1 import get_variable
from tensorflow.compat.v1.initializers import he_normal
# from tensorflow import pad
import tensorflow as tf


def conv_bn_relu(input_t,
                 out_channel,
                 name,
                 strides=1,
                 ksize=3,
                 has_bn=True,
                 has_relu=True,
                 has_bias=True):
    
    with tf.variable_scope(name):
        in_channel = input_t.shape[3]
        conv_weights = get_variable(
                            name=f'weights',
                            shape=[ksize, ksize, in_channel, out_channel],
                            initializer=he_normal(),
                            trainable=True)
        if has_bias:
            conv_bias = get_variable(
                            name=f'bias',
                            shape=[out_channel,],
                            initializer=he_normal(),
                            trainable=True)
        if has_bn:
            bn_means = get_variable(
                            name=f'means',
                            shape=[out_channel,],
                            initializer=he_normal(),
                            trainable=True)
            bn_variances = get_variable(
                            name=f'variances',
                            shape=[out_channel,],
                            initializer=he_normal(),
                            trainable=True)

        output_t = conv2d(input_t, 
                          filter=conv_weights,
                          strides=strides,
                          padding='VALID',
                          name=f'conv')
        if has_bias:
            output_t = bias_add(output_t, conv_bias, name=f'biasadd')
        if has_bn:
            output_t = batch_normalization(output_t,
                                           mean=bn_means,
                                           variance=bn_variances,
                                           offset=None,
                                           scale=None,
                                           variance_epsilon=1e-05,
                                           name=f'bn')
        if has_relu:
            output_t = relu(output_t, name=f'relu')

        return output_t






def xcorr_depthwise(x, kernel, name):
    '''
    borrow from https://github.com/torrvision/siamfc-tf/blob/master/src/siamese.py
    '''
    with tf.variable_scope(name):
        net_z = tf.transpose(kernel, perm=[1,2,0,3])
        net_x = tf.transpose(x, perm=[1,2,0,3])

        Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
        Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))

        net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
        net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
        net_final = depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID', name='correlation')

        net_final = tf.concat(tf.split(net_final, 3, axis=3), axis=0)

        net_final = tf.expand_dims(tf.reduce_sum(net_final, axis=3), axis=3)

        return net_final


