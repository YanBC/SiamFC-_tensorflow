import tensorflow.compat.v1 as tf

class Classification_Acc:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __call__(self, y_true, y_pred):
        gt = tf.math.argmax(y_true, axis=1)
        pd = tf.math.argmax(y_pred, axis=1)
        results = tf.equal(gt, pd)
        num_batch = tf.cast(self.batchsize, tf.float32)
        acc = tf.math.reduce_sum(tf.cast(results, dtype=tf.float32)) / num_batch
        return acc