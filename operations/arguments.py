import argparse

from util.types import MetricType

DATASET_ARGUMENT = {
    'name': ['--dataset', '-d'],
    'help': 'Dataset config file',
    'type': str,
    'required': True
}
NETWORK_ARGUMENT = {
    'name': ['--network', '-n'],
    'help': 'Network config file',
    'type': str,
    'required': True
}
WEIGHTS_FILE_ARGUMENT = {
    'name': ['--weights'],
    'help': 'Name of the file containing the model weights. Must either be located in the "weights" folder of the current dataset-network combo or should be the path to the file using the same network. Leave empty to use the highest scoring weights file.',
    'type': str,
    'default': None,
    'required': False
}
SAVE_MODEL_ARGUMENT = {
    'name': ['--save_model'],
    'help': 'Whether to dilate the validation labels.',
    'type': bool,
    'default': False,
    'required': False,
    'action': argparse.BooleanOptionalAction    # Enables --no-save-model
}
EPOCHS_PER_CHECKPOINT_ARGUMENT = {
    'name': ['--checkpoint_epochs'],
    'help': 'How many epochs should pass per checkpoint.',
    'type': int,
    'default': 5,
    'required': False,
}
MONITOR_METRIC_ARGUMENT = {
    'name': ['--monitor_metric'],
    'help': 'The metric to monitor in the progression plot and as the main reason improvement heuristic.',
    'type': MetricType,
    'default': MetricType.F1Score,
    'required': False,
}
NO_LABELS_ARGUMENT = {
    'name': ['--no_labels'],
    'help': 'Whether this dataset has labels. When this flag is provided, the dataset will be supplemented using empty labels.',
    'default': False,
    'required': False,
    'action': 'store_true'
}
VISUALIZE_COMPARISONS_ARGUMENT = {
    'name': ['--visualize_comparisons'],
    'help': 'Whether to visualize the predictions as a comparison between the ground truth and the predicted label.',
    'default': False,
    'required': False,
    'action': 'store_true'
}