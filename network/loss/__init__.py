from typing import Callable

import keras

from network.loss.loss_functions import weighted_binary_cross_entropy
from util.config.network_config import NetworkConfig
from util.types import LossType


FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
WCE_BETA = 10

def determine_loss_function(config: NetworkConfig) -> Callable[[keras.KerasTensor, keras.KerasTensor], keras.KerasTensor]:
    """Determine the loss function using the config and function specific values around it."""
    match config.loss:
        case LossType.FocalLoss:
            return keras.losses.BinaryFocalCrossentropy(from_logits=False, apply_class_balancing=False, alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
        case LossType.BCE:
            return keras.losses.BinaryCrossentropy()
        case LossType.WCE:
            return weighted_binary_cross_entropy(WCE_BETA)
        case LossType.Dice:
            return keras.losses.Dice()
        case _:
            raise ValueError(f'Unknown loss type: {config.loss}')
