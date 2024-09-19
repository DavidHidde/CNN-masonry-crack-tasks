from typing import Callable

import tensorflow as tf
import keras

from network.loss.loss_functions import weighted_binary_cross_entropy, dilated_dice_loss
from util.config.network_config import NetworkConfig
from util.types import LossType


FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
WCE_BETA = 10

def determine_loss_function(config: NetworkConfig) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Determine the loss function using the config and function specific values around it."""
    match config.loss:
        case LossType.FocalLoss:
            return tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=False, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
        case LossType.BCE:
            return tf.keras.losses.BinaryCrossentropy()
        case LossType.WCE:
            return weighted_binary_cross_entropy(WCE_BETA)
        case LossType.F1Score:
            return keras.losses.Dice()
        case LossType.F1ScoreDilate:
            return dilated_dice_loss
        case _:
            raise ValueError(f'Unknown loss type: {config.loss}')
