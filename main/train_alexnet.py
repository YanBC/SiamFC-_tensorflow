import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
import os
import signal
import sys
sys.path.append('.')

from networks.models import AlexNet
from data.classification_datasets import Data, Imagenet2012, Alexnet_Formater, Alexnet_Sampler
from data.dataloader import DataLoader


def classification_loss(y_true, y_pred):
    loss = -1 * y_true * tf.log(y_pred)
    return tf.math.reduce_sum(loss) / tf.cast(y_true.shape[0], tf.float32)

def SIGINT_handler(signum, frame):
    print(f'Sigal #{signum} receive. Exiting...')
    global datagen
    datagen.shutdown()
    exit(0)


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

    datagen = DataLoader(sampler, num_worker=8, buffer_size=16)
    signal.signal(signal.SIGINT, SIGINT_handler)

    save_graph_path = './temp/alexnet.pb'
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Input'):
            img_t = tf.placeholder(tf.float32, shape=(batchsize, input_size, input_size, input_channel), name='image')
            y_true_t = tf.placeholder(tf.float32, shape=(batchsize, num_cls), name='y_true')

        network = AlexNet()
        output_t = network(img_t)
        loss_t = classification_loss(y_true_t, output_t)
        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init_op)

        def train_job(lr, total_steps, report_interval=100):
            import time
            last_interval = None

            optimizer = AdamOptimizer(learning_rate=lr)
            op_minimize = optimizer.minimize(loss_t)
            sess.run(tf.variables_initializer(optimizer.variables()))
            print(f'Setting learning rate to be {lr} for the next {total_steps} steps...')
            for step in range(total_steps):
                data = datagen.load_one()
                loss, result = sess.run([loss_t, op_minimize], feed_dict={img_t: data['X'], y_true_t: data['Y']})
                if step % report_interval == 0:
                    if last_interval is None:
                        print('loss: %0.3f' % (loss))
                    else:
                        now = time.time()
                        print('loss: %0.3f    time: %0.3f steps per second' % (loss, report_interval/(now-last_interval)))
                    last_interval = time.time()
            with open(save_graph_path, 'wb') as f:
                f.write(graph.as_graph_def().SerializeToString())
            print(f"Save model to {save_graph_path}")

        # train_job(lr=0.001, total_steps=300)
        train_job(lr=0.001, total_steps=5000, report_interval=500)
        train_job(lr=0.0003, total_steps=50000, report_interval=500)
        train_job(lr=0.0001, total_steps=100000, report_interval=500)

    datagen.shutdown()


