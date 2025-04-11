from typing import Callable

import keras
import keras.backend as K
from keras import ops

from util.config import NetworkConfig
from util.types import LossType

FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
WCE_BETA = 10


def clip_sum(tensor: keras.KerasTensor) -> float:
    """Clip the values between 0 and 1 and then sum them."""
    return ops.sum(ops.clip(tensor, K.epsilon(), 1. - K.epsilon()))


@keras.saving.register_keras_serializable()
def weighted_binary_cross_entropy(beta: float) -> Callable[[keras.KerasTensor, keras.KerasTensor], keras.KerasTensor]:
    """
    Weighted Cross-Entropy (WCE) Loss.
    Applies binary cross entropy loss using a beta value for weights.

    See https://medium.com/the-owl/weighted-binary-cross-entropy-losses-in-keras-e3553e28b8db
    """

    def loss_function(y_true: keras.KerasTensor, y_pred: keras.KerasTensor) -> keras.KerasTensor:
        y_pred = ops.clip(ops.convert_to_tensor(y_pred), K.epsilon(), 1. - K.epsilon())
        y_true = ops.clip(ops.cast(y_true, y_pred.dtype), K.epsilon(), 1. - K.epsilon())

        bce = beta * y_true * ops.log(y_pred)  # Positive class, apply weight beta
        bce += (1. - y_true) * ops.log(1. - y_pred)  # Negative class, apply weight 1.
        return ops.mean(-bce, axis=-1)

    return loss_function


def determine_loss_function(config: NetworkConfig) -> Callable[
    [keras.KerasTensor, keras.KerasTensor], keras.KerasTensor]:
    """Determine the loss function using the config and function specific values around it."""
    match config.loss:
        case LossType.FocalLoss:
            return keras.losses.BinaryFocalCrossentropy(
                from_logits=False,
                apply_class_balancing=False,
                alpha=FOCAL_LOSS_ALPHA,
                gamma=FOCAL_LOSS_GAMMA
            )
        case LossType.BCE:
            return keras.losses.BinaryCrossentropy()
        case LossType.WCE:
            return weighted_binary_cross_entropy(WCE_BETA)
        case LossType.Dice:
            return keras.losses.Dice()
        case _:
            raise ValueError(f'Unknown loss type: {config.loss}')
