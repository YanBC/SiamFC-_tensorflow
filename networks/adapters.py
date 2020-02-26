from .common import conv_bn_relu
from .config import network_config as cfg




def r_z_k(input_t):
    output_t = conv_bn_relu(input_t, name='r_z_k', out_channel=cfg['head_width'], strides=1, ksize=3, has_relu=False)
    return output_t


def c_z_k(input_t):
    output_t = conv_bn_relu(input_t, name='c_z_k', out_channel=cfg['head_width'], strides=1, ksize=3, has_relu=False)
    return output_t


def r_x(input_t):
    output_t = conv_bn_relu(input_t, name='r_x', out_channel=cfg['head_width'], strides=1, ksize=3, has_relu=False)
    return output_t


def c_x(input_t):
    output_t = conv_bn_relu(input_t, name='c_x', out_channel=cfg['head_width'], strides=1, ksize=3, has_relu=False)
    return output_t