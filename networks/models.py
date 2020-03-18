import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import get_variable, constant

from .common import Conv_Bn_Relu, Max_Pooling, Fully_Connected, Xcorr_Depthwise


class ModelBase(tf.Module):
    '''
    The name argument is only here to be compatible with tf.Module,
    DO NOT initialize this class with a name. This rule also applies
    to all child classes.
    '''
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self):
        pass

    @property
    def num_trainable_parameters(self):
        if hasattr(self, '_num_trainable_parameters'):
            return self._num_trainable_parameters
        else:
            total_num = 0
            for varibale in self.trainable_variables:
                total_num += varibale.shape.num_elements()
            self._num_trainable_parameters = total_num
            return total_num
    

class AlexNet_Feat(ModelBase):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.block_1 = Conv_Bn_Relu(in_channel=3, out_channel=96, strides=2, ksize=11, name='block_1')
        self.block_2 = Conv_Bn_Relu(in_channel=96, out_channel=256, strides=1, ksize=5, name='block_2')
        self.block_3 = Conv_Bn_Relu(in_channel=256, out_channel=384, strides=1, ksize=3, name='block_3')
        self.block_4 = Conv_Bn_Relu(in_channel=384, out_channel=384, strides=1, ksize=3, name='block_4')
        self.block_5 = Conv_Bn_Relu(in_channel=384, out_channel=256, strides=1, ksize=3, has_relu=False, name='block_5')

        self.pool_1 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_1')
        self.pool_2 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool_2')

    def __call__(self, input_t):
        output_t = self.block_1(input_t)
        output_t = self.pool_1(output_t)
        output_t = self.block_2(output_t)
        output_t = self.pool_2(output_t)
        output_t = self.block_3(output_t)
        output_t = self.block_4(output_t)
        output_t = self.block_5(output_t)
        return output_t


class AlexNet(ModelBase):
    def __init__(self, name=None):
        super().__init__(name=name)

        with tf.variable_scope('Feat'):
            self.feat = AlexNet_Feat()

        self.pool_3 = Max_Pooling(ksize=3, strides=2, padding='VALID', name='pool')
        self.conv_1 = Conv_Bn_Relu(in_channel=256, out_channel=256, strides=1, ksize=3, name='conv_1')
        self.conv_2 = Conv_Bn_Relu(in_channel=256, out_channel=128, strides=1, ksize=3, name='conv_2')

        self.fc_1 = Fully_Connected(in_size=2048, out_size=2048, has_relu=True, name='fc_1')
        self.fc_2 = Fully_Connected(in_size=2048, out_size=1000, has_softmax=True, name='fc_2')

    def __call__(self, input_t):
        output_t = self.feat(input_t)

        output_t = self.pool_3(output_t)
        output_t = self.conv_1(output_t)
        output_t = self.conv_2(output_t)

        output_t = tf.layers.flatten(output_t)
        output_t = self.fc_1(output_t)
        output_t = self.fc_2(output_t)

        return output_t


class Adapter(ModelBase):
    def __init__(self, name=None, head_width=256):
        super().__init__(name=name)
        self.conv = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_relu=False, name='conv')

    def __call__(self, input_t):
        output_t = self.conv(input_t)
        return output_t


class DenseBoxHead(ModelBase):
    def __init__(self, name=None, x_size=303, z_size=127, head_width=256, total_stride=8, score_size=17):

        super().__init__(name=name)
        score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2

        self.cls_conv_1 = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_bn=False, name='cls_conv_1')
        self.cls_conv_2 = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_bn=False, name='cls_conv_2')
        self.cls_conv_3 = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_bn=True, name='cls_conv_3')

        self.reg_conv_1 = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_bn=False, name='reg_conv_1')
        self.reg_conv_2 = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_bn=False, name='reg_conv_2')
        self.reg_conv_3 = Conv_Bn_Relu(in_channel=head_width, out_channel=head_width, strides=1, ksize=3, has_bn=True, name='reg_conv_3')

        self.cls_score = Conv_Bn_Relu(in_channel=head_width, out_channel=1, strides=1, ksize=1, has_relu=False, name='cls_score_conv')

        self.ctr_score = Conv_Bn_Relu(in_channel=head_width, out_channel=1, strides=1, ksize=1, has_relu=False, name='ctr_score_conv')

        self.offset = Conv_Bn_Relu(in_channel=head_width, out_channel=4, strides=1, ksize=1, has_relu=False, name='offset_conv')

        with tf.variable_scope('fm_sb'):
            self.bi = get_variable(name='bi', initializer=0.0, trainable=True, dtype=tf.float32)
            self.si = get_variable(name='si', initializer=1.0, trainable=True, dtype=tf.float32)

        with tf.variable_scope('fm_ctr'):
            self.total_stride = constant(total_stride, dtype=tf.float32)
            self.fm_ctr = self._get_xy_ctr(score_size, score_offset, total_stride)

    def __call__(self, input_cls, input_reg):
        output_conf = input_cls
        output_bbox = input_reg

        output_conf = self.cls_conv_1(output_conf)
        output_conf = self.cls_conv_2(output_conf)
        output_conf = self.cls_conv_3(output_conf)
        output_bbox = self.reg_conv_1(output_bbox)
        output_bbox = self.reg_conv_2(output_bbox)
        output_bbox = self.reg_conv_3(output_bbox)

        cls_score = self.cls_score(output_conf)
        with tf.name_scope('cls_score'):
            cls_B, cls_H, cls_W, cls_C = tf.unstack(tf.shape(cls_score, name='shape'))
            cls_score = tf.reshape(cls_score, (cls_B, cls_H*cls_W, cls_C), name='reshape')

        ctr_score = self.ctr_score(output_conf)
        with tf.name_scope('ctr_score'):
            ctr_B, ctr_H, ctr_W, ctr_C = tf.unstack(tf.shape(ctr_score, name='shape'))
            ctr_score = tf.reshape(ctr_score, (ctr_B, ctr_H*ctr_W, ctr_C), name='reshape')

        offset = self.offset(output_bbox)
        with tf.name_scope('offset'):
            offset = (self.si * offset + self.bi) * self.total_stride
            offset = tf.exp(offset, name='exp')

            offset_B, offset_H, offset_W, offset_C = tf.unstack(tf.shape(offset, name='shape'))
            offset = tf.reshape(offset, (offset_B, offset_H*offset_W, offset_C), name='reshape')
            xy0 = self.fm_ctr - offset[:, :, 0:2]
            xy1 = self.fm_ctr + offset[:, :, 2:]
            bbox = tf.concat([xy0, xy1], axis=2)

        return cls_score, ctr_score, bbox

    def _get_xy_ctr(self, score_size, score_offset, total_stride):
        fm_height, fm_width = score_size, score_size

        y_list = tf.linspace(0., fm_height - 1., fm_height)
        x_list = tf.linspace(0., fm_width - 1., fm_width)
        X, Y = tf.meshgrid(x_list, y_list)

        XY = score_offset + tf.stack([X,Y], axis=-1) * total_stride
        XY = tf.reshape(XY, (1, fm_height*fm_width, 2))
        return XY


class SiamFCpp(ModelBase):
    def __init__(self, name=None, x_size=303, z_size=127, head_width=256, total_stride=8, score_size=17):

        super().__init__(name=name)

        with tf.variable_scope('AlexNet_Feat'):
            self.feat = AlexNet_Feat()

        with tf.variable_scope('R_Z_K'):
            self.r_z_k = Adapter(head_width=head_width)

        with tf.variable_scope('C_Z_K'):
            self.c_z_k = Adapter(head_width=head_width)

        with tf.variable_scope('R_X'):
            self.r_x = Adapter(head_width=head_width)

        with tf.variable_scope('C_X'):
            self.c_x = Adapter(head_width=head_width)
        
        self.r_xcorr = Xcorr_Depthwise(name='Reg_Xcorr')
        self.c_xcorr = Xcorr_Depthwise(name='Cls_Xcorr')

        with tf.variable_scope('Head'):
            self.head = DenseBoxHead(x_size=x_size, z_size=z_size, head_width=head_width, total_stride=total_stride, score_size=score_size)

    def __call__(self, input_x, input_z):
        feature_img = self.feat(input_x)
        cls_img = self.c_x(feature_img)
        reg_img = self.r_x(feature_img)

        feature_template = self.feat(input_z)
        cls_template = self.c_z_k(feature_template)
        reg_template = self.r_z_k(feature_template)

        r_out = self.r_xcorr(reg_img, reg_template)
        c_out = self.c_xcorr(cls_img, cls_template)

        cls_score, ctr_score, bbox = self.head(c_out, r_out)

        return cls_score, ctr_score, bbox


