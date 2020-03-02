import tensorflow as tf
if __name__ == '__main__':
    tf.enable_eager_execution()
from tensorflow.compat.v1.nn import conv2d, bias_add, batch_normalization, relu
from tensorflow.compat.v1.nn import depthwise_conv2d
from tensorflow.compat.v1 import get_variable
from tensorflow.compat.v1.initializers import he_normal



def conv_bn_relu(input_t,
                out_channel,
                name,
                strides=1,
                ksize=3,
                has_bn=True,
                has_relu=True,
                has_bias=True):
    
    with tf.variable_scope(name, initializer=he_normal(), reuse=tf.compat.v1.AUTO_REUSE):
        in_channel = input_t.shape[3]

        with tf.variable_scope('weights'):
            conv_weights = get_variable(
                                name='weights',
                                shape=[ksize, ksize, in_channel, out_channel],
                                trainable=True)
            if has_bias:
                conv_bias = get_variable(
                                name='bias',
                                shape=[out_channel,],
                                trainable=True)
            if has_bn:
                bn_means = get_variable(
                                name='means',
                                shape=[out_channel,],
                                trainable=True)
                bn_variances = get_variable(
                                name='variances',
                                shape=[out_channel,],
                                trainable=True)
                bn_offset = get_variable(
                                name='offset',
                                shape=[out_channel,],
                                trainable=True)
                bn_scale = get_variable(
                                name='scale',
                                shape=[out_channel,],
                                trainable=True)

        output_t = conv2d(input_t, 
                        filter=conv_weights,
                        strides=strides,
                        padding='VALID',
                        name='conv')
        if has_bias:
            output_t = bias_add(output_t, conv_bias, name='biasadd')
        if has_bn:
            output_t = batch_normalization(output_t,
                                           mean=bn_means,
                                           variance=bn_variances,
                                           offset=bn_offset,
                                           scale=bn_scale,
                                           variance_epsilon=1e-05,
                                           name='bn')
        if has_relu:
            output_t = relu(output_t, name='relu')

        return output_t






def xcorr_depthwise(x, kernel, name):
    with tf.variable_scope(name):
        net_z = tf.transpose(kernel, perm=[1,2,0,3])
        net_x = tf.transpose(x, perm=[1,2,0,3])

        Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
        Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))

        net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
        net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
        net_final = depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID', name='correlation')

        _, H, W, _ = tf.unstack(tf.shape(net_final))
        net_final = tf.reshape(net_final, (H, W, B, C))
        net_final = tf.transpose(net_final, perm=[2,0,1,3])

        return net_final



if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    image = np.random.rand(8, 256, 26, 26)
    template = np.random.rand(8, 256, 4, 4)

    image_p = np.transpose(image, axes=[0,2,3,1])
    template_p = np.transpose(template, axes=[0,2,3,1])
    x = tf.constant(image_p)
    z = tf.constant(template_p)

    out = xcorr_depthwise(x, z, name='xor')

