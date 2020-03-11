import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
import os
import sys
sys.path.append('.')

from networks.models import AlexNet
from data.classification_datasets import Data, Imagenet2012, Alexnet_Formater, Alexnet_Sampler
from data.dataloader import DataLoader


def classification_loss(y_true, y_pred):
    loss = -1 * y_true * tf.log(y_pred)
    return tf.math.reduce_sum(loss) / tf.cast(y_true.shape[0], tf.float32)


if __name__ == '__main__':
    imagenet_dir = './datasets/imagenet'
    dataName = 'imagenet2012.pkl'
    dataset = Imagenet2012(imagenet_dir)
    dataset.load_data_from_file(os.path.join(dataset.storage, dataName))

    input_size = 224
    input_channel = 3
    channel_mean = dataset.channel_mean
    num_cls = 1000
    formater = Alexnet_Formater(input_size, channel_mean, num_cls)

    batchsize = 32
    sampler = Alexnet_Sampler(dataset, formater, batchsize)

    datagen = DataLoader(sampler)

    total_steps = 100
    learning_rate = 0.001
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Input'):
            img_t = tf.placeholder(tf.float32, shape=(batchsize, input_size, input_size, input_channel), name='image')
            y_true_t = tf.placeholder(tf.float32, shape=(batchsize, num_cls), name='y_true')

        network = AlexNet()
        output_t = network(img_t)
        loss_t = classification_loss(y_true_t, output_t)

        optimizer = AdamOptimizer(learning_rate=learning_rate)
        op_minimize = optimizer.minimize(loss_t)

        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init_op)
        
        for step in range(total_steps):
            data = datagen.load_one()
            loss, result = sess.run([loss_t, op_minimize], feed_dict={img_t: data['X'], y_true_t: data['Y']})
            
            print(loss)
            print(result)

        datagen.shutdown()


