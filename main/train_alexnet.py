import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.train import AdamOptimizer
import os
import signal
import numpy as np
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
    dataName = 'imagenet2012_filtered.pkl'
    dataset = Imagenet2012(imagenet_dir)
    dataset.load_data_from_file(os.path.join(dataset.storage, dataName))

    input_size = 224
    input_channel = 3
    channel_mean = dataset.channel_mean
    num_cls = 1000
    formater = Alexnet_Formater(input_size, channel_mean, num_cls)

    batchsize = 16
    sampler = Alexnet_Sampler(dataset, formater, batchsize)

    datagen = DataLoader(sampler, num_worker=16, buffer_size=64)
    signal.signal(signal.SIGINT, SIGINT_handler)

    nan_graph_path = './temp/alexnet_nan.pb'
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

        def train_job(lr, total_steps, report_interval=100, save_name=None, log_file='./temp/alexnet_train.log'):
            import time
            last_interval = None

            losses = []
            optimizer = AdamOptimizer(learning_rate=lr)
            op_minimize = optimizer.minimize(loss_t)
            sess.run(tf.variables_initializer(optimizer.variables()))
            with open(log_file, 'a+') as log:
                log.write(f'Setting learning rate to be {lr} for the next {total_steps} steps...\n')
            # print(f'Setting learning rate to be {lr} for the next {total_steps} steps...')
            for step in range(total_steps):
                data = datagen.load_one()
                loss, result = sess.run([loss_t, op_minimize], feed_dict={img_t: data['X'], y_true_t: data['Y']})
                losses.append(loss)

                if loss is np.nan:
                    with open(nan_graph_path, 'wb') as f:
                        f.write(graph.as_graph_def().SerializeToString())
                    with open(log_file, 'a+') as log:
                        log.write('Loss grows to nan\n')
                        log.write(f'Save model to {nan_graph_path}\n')
                    # print(f"Save model to {nan_graph_path}")
                    return False

                if step % report_interval == 0 and len(losses) > 1:
                    if last_interval is None:
                        pass
                        # print('loss: %0.3f' % (loss))
                    else:
                        now = time.time()
                        with open(log_file, 'a+') as log:
                            log.write('loss: %0.3f    time: %0.3f steps per second\n' % (np.array(losses).mean(), report_interval/(now-last_interval)))
                        # print('loss: %0.3f    time: %0.3f steps per second' % (np.array(losses).mean(), report_interval/(now-last_interval)))
                        losses = []
                    last_interval = time.time()

            if save_name is not None:
                save_path = save_name + f'_{np.round(loss, decimals=3)}.pb'
                with open(save_path, 'wb') as f:
                    f.write(graph.as_graph_def().SerializeToString())
                with open(log_file, 'a+') as log:
                    log.write(f'Save model to {save_path}\n')
                # print(f"Save model to {save_path}")
            return True

        learning_rates = np.linspace(1e-7, 1e-4, 20)
        for i, lr in enumerate(learning_rates):
            success = train_job(lr=lr, total_steps=3000, report_interval=500)
            if not success:
                print('Loss becomes nan. Exiting...')
                break

        learning_rates = np.linspace(1e-4, 1e-6, 50)
        for i, lr in enumerate(learning_rates):
            success = train_job(lr=lr, total_steps=5000, report_interval=500, save_name=f'./temp/alexnet_{i}')
            if not success:
                print('Loss becomes nan. Exiting...')
                break

    datagen.shutdown()


