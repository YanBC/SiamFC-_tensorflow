import tensorflow.compat.v1 as tf
if __name__ == '__main__':
    tf.enable_eager_execution()



###############################
# helper functions 
###############################
def safelog(x):
    eps = np.finfo(np.float32).tiny

    return tf.log(tf.maximum(x, eps))



###############################
# loss for classification
###############################
class sigmoid_ce_retina:
    def __init__(self, weight=1.0, alpha=0.25, gamma=2.0):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, gt):
        pred = tf.sigmoid(pred)

        pos_part = (1 - pred)**self.gamma * tf.log(pred) * gt
        neg_part = pred**self.gamma * tf.log(1 - pred) * (1 - gt)
        total = -1 * (pos_part * self.alpha + neg_part * (1 - self.alpha)) 

        loss = tf.reduce_sum(total) / tf.reduce_sum(gt)
        return loss


###############################
# loss for quality assessment
###############################
class sigmoid_ce_centerness:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, pred, gt):
        # mask: object denoted by 1 and background denoted as 0
        # we only calculate the loss for object
        mask = gt
        not_neg_mask = tf.cast(pred>=0, dtype=tf.float64)

        s_pred = safelog(1.0 + tf.exp(-1 * tf.abs(pred)))
        loss = (pred * not_neg_mask - pred * gt + s_pred) * mask
        loss_residual = (-1 * gt * safelog(gt) - (1 - gt) * safelog(1 - gt)) * mask
        loss = loss - loss_residual

        loss = tf.reduce_sum(loss) / tf.reduce_sum(gt)
        return loss



###############################
# loss for bbox regression
###############################
class iou_loss:
    def __init__(self, weight=3.0):
        self.weight = weight

    def __call__(self, pred, gt, gt_cls):
        # mask: object denoted by 1 and background denoted as 0
        # we only calculate the loss for object
        mask = gt_cls[:,:,0]

        aog = (gt[:,:,2] - gt[:,:,0] + 1) * (gt[:,:,3] - gt[:,:,1] + 1)
        aop = tf.abs((pred[:,:,2] - pred[:,:,0] + 1) * (pred[:,:,3] - pred[:,:,1] + 1))

        iw = tf.minimum(pred[:, :, 2], gt[:, :, 2]) - tf.maximum(
            pred[:, :, 0], gt[:, :, 0]) + 1
        ih = tf.minimum(pred[:, :, 3], gt[:, :, 3]) - tf.maximum(
            pred[:, :, 1], gt[:, :, 1]) + 1
        inter = tf.maximum(iw, 0.0) * tf.maximum(ih, 0.0)

        union = aog + aop - inter
        iou = tf.maximum(inter / union, 0.0)
        loss = -1 * safelog(iou)

        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(gt_cls)
        return loss








##########################
# test
##########################
if __name__ == '__main__':
    import numpy as np

    B = 16
    HW = 17 * 17

    np.random.seed(0)
    pred_cls = tf.constant(np.random.rand(B, HW, 1))
    gt_cls = tf.constant(np.random.randint(2, size=(B, HW, 1)).astype(np.float64))

    pred_ctr = tf.constant(np.random.rand(B, HW, 1))
    gt_ctr = tf.constant(np.random.randint(2, size=(B, HW, 1)).astype(np.float64))

    pred_reg = tf.constant(np.random.rand(B, HW, 4))
    gt_reg = tf.constant(np.random.rand(B, HW, 4))


    cal_cls = sigmoid_ce_retina(alpha=0.5, gamma=0.0)
    cal_ctr = sigmoid_ce_centerness()
    cal_reg = iou_loss()

    loss_cls = cal_cls(pred_cls, gt_cls)
    loss_ctr = cal_ctr(pred_ctr, gt_ctr)
    loss_reg = cal_reg(pred_reg, gt_reg, gt_cls)