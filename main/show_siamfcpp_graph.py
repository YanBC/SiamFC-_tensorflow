import tensorflow.compat.v1 as tf
import numpy as np
import sys
sys.path.append('.')

from networks.models import get_siamfcpp


if __name__ == '__main__':
    graph = tf.Graph()

    with graph.as_default():
        with tf.name_scope('Input'):
            img_t = tf.placeholder(dtype=tf.float32, shape=(1, 303, 303, 3), name='image')
            template_t = tf.placeholder(dtype=tf.float32, shape=(1, 127, 127, 3), name='template')
        cls_t, ctr_t, bbox_t = get_siamfcpp(img_t, template_t)
        tf.summary.FileWriter('./temp/siamfcpp', graph=graph)

        init_op = tf.global_variables_initializer()

    sess = tf.Session(graph=graph)
    sess.run(init_op)

    np.random.seed(0)
    img = np.random.rand((1, 303, 303, 3))
    template = np.random.rand((1, 127, 127, 3))
    cls, ctr, bbox = sess.run([cls_t, ctr_t, bbox_t], feed_dict={img_t: img, template_t: template})

    with open('./temp/siamfcpp.pb', 'bw') as f:
        f.write(sess.graph.as_graph_def().SerializeToString())


    