import os
import sys
import tempfile

import tensorflow as tf
import segmentation_models as sm

from util.config import NetworkConfig
from util.types import ModelType

OUTPUT_NUM_CLASSES = 1
OUTPUT_ACTIVATION = 'sigmoid'

FPN_BLOCK_FILTERS = 512

UNET_NUM_FILTERS = 64
UNET_DECODER_FILTERS = (1024, 512, 256, 128, 64)
UNET_DECODER_BLOCK_TYPE = 'upsampling'

PSP_CONV_FILTERS = 512

LINKNET_DECODER_FILTERS = (1024, 512, 256, 128, 64)
LINKNET_DECODER_BLOCK_TYPE = 'transpose'

def add_regularization(model: tf.keras.Model, regularization: float) -> tf.keras.Model:
    """
    Add L2 regularization to a model.
	The addition of regularization is based on the implementation shown in the link:
    https://sthalles.github.io/keras-regularizer/
    """

    regularizer = tf.keras.regularizers.l2(regularization)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # Load the model from the config
    new_model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    new_model.load_weights(tmp_weights_path, by_name=True)
    return model

def build_model(network_config: NetworkConfig, input_dims: tuple[int, int, int]) -> tf.keras.Model:
    """
    Build the model itself. Note that this does not mean **compiling** the model, this still needs to be done afterwards.
    We just go past each case individually to ensure we set proper parameters.
    """
    sm_model_kwargs = {
        'input_shape': input_dims,
        'classes': OUTPUT_NUM_CLASSES,
        'activation': OUTPUT_ACTIVATION,
        'encoder_weights': 'imagenet' if network_config.use_pretrained else None,
    }
    if network_config.backbone is not None:
        sm_model_kwargs['backbone_name'] = network_config.backbone.value

    match network_config.model:
        case ModelType.DeepCrack:
            sys.path.append(os.path.join(os.getcwd(), 'libs', 'deepcrack'))
            from model import DeepCrack  # Requires Git submodule to be loaded

            model = DeepCrack(input_shape=(network_config.batch_size, *input_dims))
        case ModelType.DeepLabV3:
            raise NotImplementedError('DeepLabV3 is not yet implemented.')
        case ModelType.Unet:
            model = sm.Unet(
                decoder_filters=UNET_DECODER_FILTERS,
                decoder_block_type=UNET_DECODER_BLOCK_TYPE,
                decoder_use_batchnorm=network_config.batch_normalization,
                **sm_model_kwargs
            )
        case ModelType.PSPNet:
            model = sm.PSPNet(
                psp_conv_filters=PSP_CONV_FILTERS,
                psp_dropout=network_config.dropout,
                psp_use_batchnorm=network_config.batch_normalization,
                **sm_model_kwargs
            )
        case ModelType.FPN:
            model = sm.FPN(
                pyramid_block_filters=FPN_BLOCK_FILTERS,
                pyramid_dropout=network_config.dropout,
                **sm_model_kwargs
            )
        case ModelType.LinkNet:
            model = sm.Linknet(
                decoder_filters=LINKNET_DECODER_FILTERS,
                decoder_block_type=LINKNET_DECODER_BLOCK_TYPE,
                decoder_use_batchnorm=network_config.batch_normalization,
                **sm_model_kwargs
            )
        case _:
            raise ValueError(f'Unknown model type: {network_config.model}')

    if network_config.regularization:
        model = add_regularization(model, network_config.regularization)

    return model