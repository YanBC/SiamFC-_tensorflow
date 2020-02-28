from .backbones import alexnet
from .adapters import r_z_k, c_z_k, r_x, c_x
from .common import xcorr_depthwise
from .heads import denseboxhead
from .config import network_config as cfg


import tensorflow as tf



def get_train_model(input_x, input_z):
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

