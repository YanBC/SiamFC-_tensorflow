import tensorflow as tf

from .common import conv_bn_relu
from .config import network_config as cfg


head_width = cfg['head_width']


def denseboxhead(input_cls, input_reg):
    output_cls = input_cls
    output_bbox = input_reg

    with tf.name_scope('DenseBox') as scope:
        '''
        Convolutions
        '''
        output_cls = conv_bn_relu(output_cls, name='cls_conv3x3_1', out_channel=head_width, strides=1, ksize=3, has_bn=False)
        output_cls = conv_bn_relu(output_cls, name='cls_conv3x3_2', out_channel=head_width, strides=1, ksize=3, has_bn=False)
        output_cls = conv_bn_relu(output_cls, name='cls_conv3x3_3', out_channel=head_width, strides=1, ksize=3, has_bn=True)

        output_bbox = conv_bn_relu(output_bbox, name='reg_conv3x3_1', out_channel=head_width, strides=1, ksize=3, has_bn=False)
        output_bbox = conv_bn_relu(output_bbox, name='reg_conv3x3_2', out_channel=head_width, strides=1, ksize=3, has_bn=False)
        output_bbox = conv_bn_relu(output_bbox, name='reg_conv3x3_3', out_channel=head_width, strides=1, ksize=3, has_bn=True)

        '''
        Outputs
        '''
        with tf.name_scope('cls_score'):
            cls_score = conv_bn_relu(output_cls, name='conv', out_channel=1, strides=1, ksize=1, has_relu=False)
            cls_B, cls_H, cls_W, cls_C = tf.shape(cls_score, name='shape')
            cls_score = tf.reshape(cls_score, (cls_B, cls_H*cls_W, cls_C), name='reshape')

        with tf.name_scope('ctr_score'):
            ctr_score = conv_bn_relu(output_cls, name='conv', out_channel=1, strides=1, ksize=1, has_relu=False)
            ctr_B, ctr_H, ctr_W, ctr_C = tf.shape(ctr_score, name='shape')
            ctr_score = tf.reshape(ctr_score, (ctr_B, ctr_H*ctr_W, ctr_C), name='reshape')

        with tf.name_scope('offset'):
            offset = conv_bn_relu(output_bbox, name='conv', out_channel=4, strides=1, ksize=1, has_relu=False)
            offset = tf.exp(offset, name='exp')
            
            # ToDo
            # bbox decoding

