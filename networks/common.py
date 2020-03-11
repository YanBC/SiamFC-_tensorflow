import tensorflow.compat.v1 as tf
if __name__ == '__main__':
    tf.enable_eager_execution()
from tensorflow.compat.v1.nn import conv2d, bias_add, batch_normalization, max_pool2d
from tensorflow.compat.v1.nn import softmax, relu
from tensorflow.compat.v1.nn import depthwise_conv2d
from tensorflow.compat.v1 import get_variable
from tensorflow.compat.v1.initializers import he_normal

from tensorflow.compat.v1.layers import BatchNormalization


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



class Conv_Bn_Relu(tf.Module):
    def __init__(self, in_channel, out_channel, name, strides=1, ksize=3, has_bn=True, has_relu=True, has_bias=True):
        super().__init__(name=name)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.has_relu = has_relu
        self.has_bias = has_bias
        self.has_bn = has_bn
        self.strides = strides

        with tf.variable_scope('Weights', initializer=he_normal()):
            with tf.variable_scope(self.name):
                self.conv_weights = get_variable(name='weights', shape=[ksize, ksize, in_channel, out_channel], trainable=True)

                if self.has_bias:
                    self.conv_bias = get_variable(name='bias', shape=[out_channel,], trainable=True)

                # borrow from https://github.com/udacity/deep-learning/blob/master/batch-norm/Batch_Normalization_Solutions.ipynb
                if self.has_bn:
                    self.epsilon = 1e-3
                    self.gamma = get_variable(name='means', initializer=tf.ones([out_channel,]), trainable=True)
                    self.beta = get_variable(name='variances', initializer=tf.zeros([out_channel,]), trainable=True)
                    self.pop_mean = get_variable(name='offset', initializer=tf.zeros([out_channel,]), trainable=False)
                    self.pop_variance = get_variable(name='scale', initializer=tf.ones([out_channel,]), trainable=False)

    @tf.Module.with_name_scope
    def _batch_norm_training(self, input_t):
        batch_mean, batch_variance = tf.nn.moments(input_t, [0,1,2], keep_dims=False)

        decay = 0.99
        train_mean = tf.assign(self.pop_mean, self.pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(self.pop_variance, self.pop_variance * decay + batch_variance * (1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(input_t, batch_mean, batch_variance, self.beta, self.gamma, self.epsilon)

    @tf.Module.with_name_scope
    def _batch_norm_inference(self, input_t):
        return tf.nn.batch_normalization(input_t, self.pop_mean, self.pop_variance, self.beta, self.gamma, self.epsilon)


    @tf.Module.with_name_scope
    def __call__(self, input_t, is_training=True):
        assert input_t.shape[3] == self.in_channel

        output_t = conv2d(input_t, filter=self.conv_weights, strides=self.strides, padding='VALID', name='conv')

        if self.has_bias:
            output_t = bias_add(output_t, self.conv_bias, name='biasadd')

        if self.has_bn:
            output_t = tf.cond(tf.constant(is_training, dtype=tf.bool), lambda: self._batch_norm_training(output_t), lambda: self._batch_norm_inference(output_t))
            # # output_t = batch_normalization(output_t, mean=self.bn_means, variance=self.bn_variances, offset=self.bn_offset, scale=self.bn_scale, variance_epsilon=1e-05, name='bn')
            # self.batchnorm(output_t)

        if self.has_relu:
            output_t = relu(output_t, name='relu')

        return output_t


class Fully_Connected(tf.Module):
    def __init__(self, in_size, out_size, name, has_bias=True, has_relu=False, has_softmax=False):
        super().__init__(name=name)

        assert (has_relu == False or has_softmax == False)
        self.in_size = in_size
        self.out_size = out_size
        self.has_bias = has_bias
        self.has_relu = has_relu
        self.has_softmax = has_softmax

        with tf.variable_scope('Weights', initializer=he_normal()):
            with tf.variable_scope(self.name):
                self.weights = tf.get_variable('weights', shape=(self.in_size, self.out_size), trainable=True)
                if self.has_bias:
                    self.bias = tf.get_variable('bias', shape=(self.out_size,), trainable=True)

    @tf.Module.with_name_scope
    def __call__(self, input_t):
        output_t = tf.matmul(input_t, self.weights, name='matmul')
        if self.has_bias:
            output_t = bias_add(output_t, self.bias, name='biasadd')
        if self.has_relu:
            output_t = relu(output_t, name='relu')
        if self.has_softmax:
            output_t = softmax(output_t, name='softmax')

        return output_t


class Xcorr_Depthwise(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.Module.with_name_scope
    def __call__(self, x, kernel):
        net_z = tf.transpose(kernel, perm=[1,2,0,3])
        net_x = tf.transpose(x, perm=[1,2,0,3])

        Hz, Wz, B, C = tf.unstack(tf.shape(net_z))
        Hx, Wx, Bx, Cx = tf.unstack(tf.shape(net_x))

        net_z = tf.reshape(net_z, (Hz, Wz, B*C, 1))
        net_x = tf.reshape(net_x, (1, Hx, Wx, B*C))
        net_final = depthwise_conv2d(net_x, net_z, strides=[1,1,1,1], padding='VALID')

        _, H, W, _ = tf.unstack(tf.shape(net_final))
        net_final = tf.reshape(net_final, (H, W, B, C))
        net_final = tf.transpose(net_final, perm=[2,0,1,3])

        return net_final


class Max_Pooling(tf.Module):
    def __init__(self, name=None, ksize=3, strides=2, padding='VALID'):
        super().__init__(name=name)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    @tf.Module.with_name_scope
    def __call__(self, input_t):
        output_t = max_pool2d(input_t, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return output_t
