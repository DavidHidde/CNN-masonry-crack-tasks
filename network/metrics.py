from typing import Callable

import keras
import keras.backend as K
from keras import ops

from util.types import MetricType


def clip_sum(tensor: keras.KerasTensor) -> float:
    """Clip the values between 0 and 1 and then sum them."""
    return ops.sum(ops.clip(tensor, K.epsilon(), 1. - K.epsilon()))


@keras.saving.register_keras_serializable()
def recall(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """Recall = TP / (TP + FN)."""
    tp = clip_sum(y_true * y_pred)
    fn = clip_sum(y_true - y_pred)

    return tp / (tp + fn + K.epsilon())


@keras.saving.register_keras_serializable()
def precision(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """Precision = TP / (TP + FP)"""
    tp = clip_sum(y_true * y_pred)
    predicted_positives = clip_sum(y_pred)
    return tp / (predicted_positives + K.epsilon())


@keras.saving.register_keras_serializable()
def f1_score(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> float:
    """F1-score = 2TP / (2TP + FP + FN)."""
    tp = clip_sum(y_true * y_pred)
    fp = clip_sum(y_pred - y_true)
    fn = clip_sum(y_true - y_pred)

    return (2. * tp) / (2. * tp + fp + fn + K.epsilon())


def get_standard_metrics() -> list[Callable[[keras.KerasTensor, keras.KerasTensor], float]]:
    """Get the standard set of metrics to use."""
    return [
        keras.metrics.BinaryAccuracy(name=MetricType.Accuracy.value),
        keras.metrics.BinaryIoU(name=MetricType.IoU.value, target_class_ids=[1]),
        recall,
        precision,
        f1_score
    ]
