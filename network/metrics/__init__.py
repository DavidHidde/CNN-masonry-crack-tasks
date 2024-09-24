from typing import Callable

import keras
import tensorflow as tf

from network.metrics.metric_functions import precision, recall, f1_score, precision_dilated, f1_score_dilated


def get_standard_metrics() -> list[Callable[[tf.Tensor, tf.Tensor], float]]:
    """Get the standard set of metrics to use."""
    return [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        recall,
        precision,
        precision_dilated,
        f1_score,
        f1_score_dilated
    ]