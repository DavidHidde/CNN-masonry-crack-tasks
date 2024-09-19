import tensorflow as tf
import tensorflow.keras.backend as K
from keras.src import ops

from util.image_operations import dilation2d

DILATION_KERNEL_SIZE = 5

def clip_sum(tensor: tf.Tensor) -> float:
    """Clip the values between 0 and 1 and then sum them."""
    return ops.sum(tf.clip_by_value(tensor, K.epsilon() , 1. - K.epsilon()))

def recall(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    Recall = TP / (TP + FN). Dilate label if requested.
    """
    tp = clip_sum(dilation2d(y_true, DILATION_KERNEL_SIZE) * y_pred if dilate else y_true * y_pred)
    fn = clip_sum(y_true - y_pred)

    return tp / (tp + fn + K.epsilon())

def precision(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    Precision = TP / (TP + FP). Dilate label if requested.
    """
    if dilate:
        y_true = dilation2d(y_true, DILATION_KERNEL_SIZE)

    tp = clip_sum(y_true * y_pred)
    predicted_positives = clip_sum(y_pred)
    return tp / (predicted_positives + K.epsilon())

def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    F1-score = 2TP / (2TP + FP + FN). Dilate label if requested.
    """
    if dilate:
        y_true_dilated = dilation2d(y_true, DILATION_KERNEL_SIZE)
        tp = clip_sum(y_true_dilated * y_pred)
        fp = clip_sum(y_pred - y_true_dilated)
    else:
        tp = clip_sum(y_true * y_pred)
        fp = clip_sum(y_pred - y_true)

    fn = clip_sum(y_true - y_pred)

    return (2. * tp) / (2. * tp + fp + fn + K.epsilon())

# Macros for Keras metrics. Don't use the functions aside from that
def precision_dilated(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Keras macro"""
    return precision(y_true, y_pred, True)

def f1_score_dilated(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Keras macro"""
    return f1_score(y_true, y_pred, True)
