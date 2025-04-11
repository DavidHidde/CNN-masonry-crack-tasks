import os
import sys

from keras.models import model_from_json
from keras import Model

from util.config import NetworkConfig, OutputConfig
from util.types import ModelType


def load_model(network_config: NetworkConfig, output_config: OutputConfig, input_dims: tuple[int, int, int], weights_file: str) -> Model:
    """
    Load the model indicated in the config file.
    If none are indicated, take the best performing model for the current dataset-network combination.
    """
    # Resolve missing target by taking best performing model from the weights and model folder.
    if weights_file is None:
        candidates = [weight for weight in os.listdir(output_config.best_models_dir) if weight.endswith('.weights.h5')]
        best_value = '0000000000000'
        for candidate in candidates:
            # Assume the score is listed at the end (*_score.weights.h5)
            if float(candidate[-16:-11]) > float(best_value[-16:-11]):
                best_value = candidate

        if best_value != '0000000000000':
            weights_file = best_value
        else:
            raise ValueError('Please provide a valid model file for the current network or train a model using the current dataset-network combination.')

    # If the weights file is not a path, prefix the weights directory to it.
    if not os.sep in weights_file:
        if os.path.exists(os.path.join(output_config.best_models_dir, weights_file)):
            weights_file = os.path.join(output_config.best_models_dir, weights_file)
        elif os.path.exists(os.path.join(output_config.checkpoints_dir, weights_file)):
            weights_file = os.path.join(output_config.checkpoints_dir, weights_file)

    if not os.path.exists(weights_file) or output_config.network_dir not in weights_file:
        raise ValueError('Please provide a valid model file for the current network or train a model using the current dataset-network combination.')

    print(f'Using model {output_config.model_file} and {weights_file} for weights.')

    # Load the model file. Some model types are exceptions, but generally we can simply load a JSON.
    if network_config.model == ModelType.DeepCrack:
        sys.path.append(os.path.join(os.getcwd(), 'libs', 'deepcrack'))
        from model import DeepCrack  # Requires Git submodule to be loaded

        model = DeepCrack(input_shape=(network_config.batch_size, *input_dims))
        model.load_weights(weights_file)
        return model

    if network_config.model == ModelType.DeepLabV3:
        raise NotImplementedError('DeepLabV3 is not yet implemented.')

    # Open the file and load the model and weights
    with open(output_config.model_file, 'r') as model_json_file:
        model = model_from_json(model_json_file.read())
    model.load_weights(weights_file)

    return model