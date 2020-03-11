from .backbones import alexnet
from .adapters import r_z_k, c_z_k, r_x, c_x
from .common import xcorr_depthwise
from .heads import denseboxhead
from .config import network_config as cfg

import tensorflow.compat.v1 as tf

from .backbones import Alexnet_Feature
from .common import Conv_Bn_Relu, Max_Pooling, Fully_Connected
from tensorflow.compat.v1.initializers import he_normal



def get_siamfcpp(input_x, input_z):
    with tf.name_scope('Image'):
        feature_img = alexnet(input_x)
        cls_img = c_x(feature_img)
        reg_img = r_x(feature_img)

    with tf.name_scope('Template'):
        feature_template = alexnet(input_z)
        cls_template = c_z_k(feature_template)
        reg_template = r_z_k(feature_template)

    c_out = xcorr_depthwise(cls_img, cls_template, name='Cls_Xcorr')
    r_out = xcorr_depthwise(reg_img, reg_template, name='Reg_Xcorr')

    cls_score, ctr_score, bbox = denseboxhead(c_out, r_out)

    return cls_score, ctr_score, bbox

def get_alexnet(input_t):
    from .common import conv_bn_relu

    assert input_t.shape[1:4] == (224, 224, 3)
    feature_img = alexnet(input_t)

    with tf.variable_scope('Ad-Hoc', initializer=tf.initializers.he_normal()):
        output_t = tf.nn.max_pool2d(feature_img, ksize=3, strides=2, padding='VALID')
        output_t = conv_bn_relu(output_t, name='block_1', out_channel=256, strides=1, ksize=3)
        output_t = conv_bn_relu(output_t, name='block_2', out_channel=128, strides=1, ksize=3)
        output_t = tf.layers.flatten(output_t)

        fc1_weights = tf.get_variable('fc1_weights', shape=(2048, 2048), trainable=True)
        fc2_weights = tf.get_variable('fc2_weights', shape=(2048, 1000), trainable=True)

        output_t = tf.matmul(output_t, fc1_weights)
        output_t = tf.nn.relu(output_t)
        output_t = tf.matmul(output_t, fc2_weights)
        output_t = tf.nn.softmax(output_t)


    return output_t


class AlexNet(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.block_1 = Conv_Bn_Relu(in_channel=3, out_channel=96, strides=2, ksize=11, name='block_1')
        self.block_2 = Conv_Bn_Relu(in_channel=96, out_channel=256, strides=1, ksize=5, name='block_2')
        self.block_3 = Conv_Bn_Relu(in_channel=256, out_channel=384, strides=1, ksize=3, name='block_3')
        self.block_4 = Conv_Bn_Relu(in_channel=384, out_channel=384, strides=1, ksize=3, name='block_4')
        self.block_5 = Conv_Bn_Relu(in_channel=384, out_channel=256, strides=1, ksize=3, has_relu=False, name='block_5')

        self.pool_1 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_1')
        self.pool_2 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_2')

        self.pool_3 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_3')
        self.conv_1 = Conv_Bn_Relu(in_channel=256, out_channel=256, strides=1, ksize=3, name='conv_1')
        self.conv_2 = Conv_Bn_Relu(in_channel=256, out_channel=128, strides=1, ksize=3, name='conv_2')

        self.fc_1 = Fully_Connected(in_size=2048, out_size=2048, has_relu=True, name='fc_1')
        self.fc_2 = Fully_Connected(in_size=2048, out_size=1000, has_softmax=True, name='fc_2')

    def __call__(self, input_t):
        output_t = self.block_1(input_t)
        output_t = self.pool_1(output_t)
        output_t = self.block_2(output_t)
        output_t = self.pool_2(output_t)
        output_t = self.block_3(output_t)
        output_t = self.block_4(output_t)
        output_t = self.block_5(output_t)

        output_t = self.pool_3(output_t)
        output_t = self.conv_1(output_t)
        output_t = self.conv_2(output_t)

        output_t = tf.layers.flatten(output_t)
        output_t = self.fc_1(output_t)
        output_t = self.fc_2(output_t)

        return output_t


