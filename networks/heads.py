import tensorflow as tf
from tensorflow.compat.v1 import get_variable, constant

from .common import conv_bn_relu
from .config import network_config as cfg

XSIZE = cfg['x_size']
HEADWIDTH = cfg['head_width']
TOTALSTRIDE = cfg['head_total_stride']
SCORESIZE = cfg['head_score_size']
SCOREOFFSET = (XSIZE - 1 - (SCORESIZE - 1) * TOTALSTRIDE) // 2


def get_xy_ctr(score_size, score_offset, total_stride):
    fm_height, fm_width = score_size, score_size

    y_list = tf.linspace(0., fm_height - 1., fm_height)
    x_list = tf.linspace(0., fm_width - 1., fm_width)
    X, Y = tf.meshgrid(x_list, y_list)

    XY = score_offset + tf.stack([X,Y], axis=-1) * total_stride
    XY = tf.reshape(XY, (1, fm_height*fm_width, 2))
    return XY



def denseboxhead(input_cls, input_reg):
    output_cls = input_cls
    output_bbox = input_reg

    with tf.variable_scope('DenseBox'):
        with tf.variable_scope('constants'):
            bi = get_variable(name='bi', initializer=0.0, trainable=True, dtype=tf.float32)
            si = get_variable(name='si', initializer=1.0, trainable=True, dtype=tf.float32)
            total_stride = constant(float(TOTALSTRIDE))
            fm_ctr = get_xy_ctr(SCORESIZE, SCOREOFFSET, TOTALSTRIDE)

        with tf.variable_scope('conv'):
            output_cls = conv_bn_relu(output_cls, name='cls_conv3x3_1', out_channel=HEADWIDTH, strides=1, ksize=3, has_bn=False)
            output_cls = conv_bn_relu(output_cls, name='cls_conv3x3_2', out_channel=HEADWIDTH, strides=1, ksize=3, has_bn=False)
            output_cls = conv_bn_relu(output_cls, name='cls_conv3x3_3', out_channel=HEADWIDTH, strides=1, ksize=3, has_bn=True)

            output_bbox = conv_bn_relu(output_bbox, name='reg_conv3x3_1', out_channel=HEADWIDTH, strides=1, ksize=3, has_bn=False)
            output_bbox = conv_bn_relu(output_bbox, name='reg_conv3x3_2', out_channel=HEADWIDTH, strides=1, ksize=3, has_bn=False)
            output_bbox = conv_bn_relu(output_bbox, name='reg_conv3x3_3', out_channel=HEADWIDTH, strides=1, ksize=3, has_bn=True)

        with tf.variable_scope('cls_score'):
            cls_score = conv_bn_relu(output_cls, name='conv', out_channel=1, strides=1, ksize=1, has_relu=False)
            cls_B, cls_H, cls_W, cls_C = tf.unstack(tf.shape(cls_score, name='shape'))
            cls_score = tf.reshape(cls_score, (cls_B, cls_H*cls_W, cls_C), name='reshape')

        with tf.variable_scope('ctr_score'):
            ctr_score = conv_bn_relu(output_cls, name='conv', out_channel=1, strides=1, ksize=1, has_relu=False)
            ctr_B, ctr_H, ctr_W, ctr_C = tf.unstack(tf.shape(ctr_score, name='shape'))
            ctr_score = tf.reshape(ctr_score, (ctr_B, ctr_H*ctr_W, ctr_C), name='reshape')

        with tf.variable_scope('offset'):
            offset = conv_bn_relu(output_bbox, name='conv', out_channel=4, strides=1, ksize=1, has_relu=False)
            offset = (si * offset + bi) * total_stride
            offset = tf.exp(offset, name='exp')

            offset_B, offset_H, offset_W, offset_C = tf.unstack(tf.shape(offset, name='shape'))
            offset = tf.reshape(offset, (offset_B, offset_H*offset_W, offset_C), name='reshape')
            xy0 = fm_ctr - offset[:, :, 0:2]
            xy1 = fm_ctr + offset[:, :, 2:]
            bbox = tf.concat([xy0, xy1], axis=2)

        return cls_score, ctr_score, bbox






if __name__ == '__main__':
    tmp = get_xy_ctr(SCORESIZE, SCOREOFFSET, TOTALSTRIDE)

