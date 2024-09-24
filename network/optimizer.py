from keras.optimizers import Adam, SGD, RMSprop
from keras.src.optimizers import optimizer

from util.config.network_config import NetworkConfig
from util.types import OptimizerType


def determine_optimizer(config: NetworkConfig) -> optimizer.Optimizer:
    """Determine the optimizer from the config. For Adam it is recommended to run the legacy version on Apple silicon."""
    match config.optimizer:
        case OptimizerType.Adam:
            return Adam(config.initial_learning_rate)
        case OptimizerType.SGD:
            return SGD(config.initial_learning_rate)
        case OptimizerType.RMSprop:
            return RMSprop(config.initial_learning_rate)
        case _:
            raise ValueError(f'Unknown loss type: {config.loss}')
