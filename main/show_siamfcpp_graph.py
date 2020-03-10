import tensorflow.compat.v1 as tf
import sys
sys.path.append('.')

from networks.models import get_siamfcpp


if __name__ == '__main__':
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('Input'):
            img_t = tf.random.normal(shape=(1, 303, 303, 3), name='image')
            template_t = tf.random.normal(shape=(1, 127, 127, 3), name='template')
        cls_t, ctr_t, bbox_t = get_siamfcpp(img_t, template_t)

    tf.summary.FileWriter('./temp/train_model', graph=graph)