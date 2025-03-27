from typing import Callable

import tensorflow as tf
import tensorflow.keras.backend as K
from keras import ops

from util.image_operations import dilation2d
from util.types import MetricType

DILATION_KERNEL_SIZE = 5
MASK_THRESHOLD = 0.5


def clip_sum(tensor: tf.Tensor) -> float:
    """Clip the values between 0 and 1 and then sum them."""
    return ops.sum(ops.clip(tensor, 0. + K.epsilon(), 1.))


def get_confusion_matrix(y_true: tf.Tensor, y_pred: tf.Tensor, dilate: bool = False) -> tuple[float, float, float, float]:
    """Get the confusion matrix from the predictions and ground truth. Dilate if necessary."""
    y_pred = ops.where(y_pred > MASK_THRESHOLD, 1., 0.)

    if dilate:
        y_true_dilated = dilation2d(y_true, DILATION_KERNEL_SIZE)
        tp = clip_sum(y_true_dilated * y_pred)
        fp = clip_sum(y_pred - y_true_dilated)
    else:
        tp = clip_sum(y_true * y_pred)
        fp = clip_sum(y_pred - y_true)

    tn = clip_sum(y_true * -1 * (1 - y_pred))
    fn = clip_sum(y_true - y_pred)

    return tp, fp, tn, fn


def recall(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    Recall = TP / (TP + FN). Dilate label if requested.
    """
    tp, _, _, fn = get_confusion_matrix(y_true, y_pred, dilate=dilate)
    return tp / (tp + fn)


def precision(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    Precision = TP / (TP + FP). Dilate label if requested.
    """
    tp, fp, _, _ = get_confusion_matrix(y_true, y_pred, dilate=dilate)
    return tp / (tp + fp)


def f1_score(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    F1-score = 2TP / (2TP + FP + FN). Dilate label if requested.
    """
    tp, fp, _, fn = get_confusion_matrix(y_true, y_pred, dilate=dilate)
    return (2. * tp) / (2. * tp + fp + fn)


def iou(y_true: tf.Tensor, y_pred: tf.Tensor, dilate=False) -> float:
    """
    IoU = TP / (TP + FP + FN). Dilate label if requested.
    """
    tp, fp, _, fn = get_confusion_matrix(y_true, y_pred, dilate=dilate)
    return tp / (tp + fp + fn)


class MetricWrapper(tf.keras.metrics.Metric):
    """Metric class for wrapping simple functions into Keras metrics."""

    fn: Callable
    dilate: bool

    def __init__(self, fn: Callable, dilate: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.variable = self.add_weight(initializer='zeros', name=fn.__name__)
        self.num_batches = self.add_weight(initializer='zeros', name='num_batches')
        self.dilate = dilate

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Update variable value"""
        self.num_batches.assign_add(1)
        self.variable.assign_add(self.fn(y_true, y_pred, self.dilate))

    def result(self) -> float:
        return self.variable / self.num_batches


def get_standard_metrics() -> list[Callable[[tf.Tensor, tf.Tensor], float]]:
    """Get the standard set of metrics to use."""
    return [
        tf.keras.metrics.BinaryAccuracy(name=MetricType.Accuracy.value, threshold=MASK_THRESHOLD),
        MetricWrapper(iou, dilate=False, name=MetricType.IoU.value),
        MetricWrapper(iou, dilate=True, name=MetricType.IoUDilated.value),
        MetricWrapper(recall, dilate=False, name=MetricType.Recall.value),
        MetricWrapper(precision, dilate=False, name=MetricType.Precision.value),
        MetricWrapper(precision, dilate=True, name=MetricType.PrecisionDilated.value),
        MetricWrapper(f1_score, dilate=False, name=MetricType.F1Score.value),
        MetricWrapper(f1_score, dilate=True, name=MetricType.F1ScoreDilated.value),
    ]
