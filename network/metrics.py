import sys
from typing import Callable

import keras
import keras.backend as K
from keras import ops

from util.types import MetricType

DILATION_KERNEL_SIZE = 5
MASK_THRESHOLD = 0.5


def clip_sum(tensor: keras.KerasTensor) -> float:
    """Clip the values between 0 and 1 and then sum them."""
    return ops.sum(ops.clip(tensor, 0. + K.epsilon(), 1.))


def get_confusion_matrix(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> tuple[float, float, float, float]:
    """Get the confusion matrix from the predictions and ground truth. Dilate if necessary."""
    y_pred = ops.where(y_pred > MASK_THRESHOLD, 1., 0.)

    tp = clip_sum(y_true * y_pred)
    fp = clip_sum(y_pred - y_true)
    tn = clip_sum(y_true * -1 * (1 - y_pred))
    fn = clip_sum(y_true - y_pred)

    return tp, fp, tn, fn


def recall(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """
    Recall = TP / (TP + FN). Dilate label if requested.
    """
    tp, _, _, fn = get_confusion_matrix(y_true, y_pred)
    return tp / (tp + fn)


def precision(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """
    Precision = TP / (TP + FP). Dilate label if requested.
    """
    tp, fp, _, _ = get_confusion_matrix(y_true, y_pred)
    return tp / (tp + fp)


def f1_score(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """
    F1-score = 2TP / (2TP + FP + FN). Dilate label if requested.
    """
    tp, fp, _, fn = get_confusion_matrix(y_true, y_pred)
    return (2. * tp) / (2. * tp + fp + fn)


def iou(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """
    IoU = TP / (TP + FP + FN). Dilate label if requested.
    """
    tp, fp, _, fn = get_confusion_matrix(y_true, y_pred)
    return tp / (tp + fp + fn)


@keras.saving.register_keras_serializable()
class MetricWrapper(keras.metrics.Metric):
    """Metric class for wrapping simple functions into Keras metrics."""

    fn: Callable

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fn = fn
        self.variable = self.add_variable(shape=(), initializer='zeros', name=fn.__name__)
        self.num_batches = self.add_variable(shape=(), initializer='zeros', name='num_batches')

    def update_state(self, y_true: keras.KerasTensor, y_pred: keras.KerasTensor, sample_weight=None):
        """Update variable value"""
        self.num_batches.assign_add(1)
        self.variable.assign_add(self.fn(y_true, y_pred))

    def result(self) -> float:
        """Return current value."""
        return self.variable / self.num_batches

    def get_config(self) -> dict[str, str]:
        """Serialize to a dict"""
        return {'fn': self.fn.__name__}

    @classmethod
    def from_config(cls, config):
        """Initialize from a config."""
        fn_name = config.pop('fn')
        current_module = sys.modules[__name__]
        return cls(getattr(current_module, fn_name), **config)


def get_standard_metrics() -> list[MetricWrapper]:
    """Get the standard set of metrics to use."""
    return [
        keras.metrics.BinaryAccuracy(name=MetricType.Accuracy.value, threshold=MASK_THRESHOLD),
        MetricWrapper(iou, name=MetricType.IoU.value),
        MetricWrapper(recall, name=MetricType.Recall.value),
        MetricWrapper(precision, name=MetricType.Precision.value),
        MetricWrapper(f1_score, name=MetricType.F1Score.value),
    ]
