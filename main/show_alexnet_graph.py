import tensorflow.compat.v1 as tf
import sys
sys.path.append('.')

from networks.models import get_alexnet


# if __name__ == '__main__':
#     graph = tf.Graph()

#     with graph.as_default():
#         with tf.name_scope('Input'):
#             img_t = tf.random.normal(shape=(1, 224, 224, 3), name='image')
#         output_t = get_alexnet(img_t)

#     tf.summary.FileWriter('./temp/alextnet', graph=graph)

from networks.models import AlexNet

if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Input'):
            img_t = tf.random.normal(shape=(1, 224, 224, 3), name='image')
        network = AlexNet()
        output_t = network(img_t)

    tf.summary.FileWriter('./temp/alextnet', graph=graph)