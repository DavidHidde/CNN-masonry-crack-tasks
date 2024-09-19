from typing import Callable

import tensorflow as tf
import tensorflow.keras.backend as K
from keras.src import ops

from util.image_operations import dilation2d

def clip_sum(tensor: tf.Tensor) -> float:
    """Clip the values between 0 and 1 and then sum them."""
    return ops.sum(tf.clip_by_value(tensor, K.epsilon() , 1. - K.epsilon()))

def weighted_binary_cross_entropy(beta: float) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted Cross-Entropy (WCE) Loss.
    Applies binary cross entropy loss using a beta value for weights.

    See https://medium.com/the-owl/weighted-binary-cross-entropy-losses-in-keras-e3553e28b8db
    """
    def loss_function(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(ops.convert_to_tensor(y_pred), K.epsilon(), 1. - K.epsilon())
        y_true = tf.clip_by_value(ops.cast(y_true, y_pred.dtype), K.epsilon(), 1. - K.epsilon())

        bce = beta * y_true * tf.math.log(y_pred)         # Positive class, apply weight beta
        bce += (1. - y_true) * tf.math.log(1. - y_pred)   # Negative class, apply weight 1.
        return ops.mean(-bce, axis=-1)

    return loss_function

def dilated_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice loss (f1-loss) but dilated.
    We need to keep in mind that the dilation should only affect TPs and FPs, so the usual trick of summing y_true and y_pred doesn't work.

    See https://github.com/keras-team/keras/blob/v3.3.3/keras/src/losses/losses.py#L1983
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)
    y_true_dilated = dilation2d(y_true, 5)

    inputs = ops.reshape(y_true, [-1])
    inputs_dilated = ops.reshape(y_true_dilated, [-1])
    targets = ops.reshape(y_pred, [-1])

    tp = clip_sum(targets * inputs_dilated)
    fp = clip_sum(targets - inputs_dilated)
    fn = clip_sum(inputs - targets)

    dice = ops.divide(
        2. * tp,
        2. * tp + fp + fn + K.epsilon(),
    )

    return 1. - dice
