import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.nn import max_pool2d

from .common import conv_bn_relu
from .common import Conv_Bn_Relu, Max_Pooling


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



class Alexnet_Feature(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.block_1 = Conv_Bn_Relu(in_channel=3, out_channel=96, strides=2, ksize=11, name='block_1')
        self.block_2 = Conv_Bn_Relu(in_channel=96, out_channel=256, strides=1, ksize=5, name='block_2')
        self.block_3 = Conv_Bn_Relu(in_channel=256, out_channel=384, strides=1, ksize=3, name='block_3')
        self.block_4 = Conv_Bn_Relu(in_channel=384, out_channel=384, strides=1, ksize=3, name='block_4')
        self.block_5 = Conv_Bn_Relu(in_channel=384, out_channel=256, strides=1, ksize=3, has_relu=False, name='block_5')

        self.pool_1 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_1')
        self.pool_2 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_2')
    
    @tf.Module.with_name_scope
    def __call__(self, input_t):
        output_t = self.block_1(input_t)
        output_t = self.pool_1(output_t)
        output_t = self.block_2(output_t)
        output_t = self.pool_2(output_t)
        output_t = self.block_3(output_t)
        output_t = self.block_4(output_t)
        output_t = self.block_5(output_t)

        return output_t