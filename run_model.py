import os

os.environ['TF_USE_LEGACY_KERAS'] = '1' # Fall back to Keras 2. We need this before loading **anything** else
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import argparse

from operations import get_operation
from util.types import OperationType


def run_model():
    """Entrypoint of the program"""
    # Gather user input
    parser = argparse.ArgumentParser(
        prog='CNN-masonry-crack-tasks',
        description='Train or evaluate crack segmentation models',
    )
    parser.add_argument('--operation', '-o', help='The operation to perform (see operations/implementations)', type=OperationType, required=True)
    args, unknown = parser.parse_known_args()

    operation = get_operation(args.operation)
    for arg_specification in operation.get_cli_arguments():
        name = arg_specification['name']
        del(arg_specification['name'])
        parser.add_argument(*name, **arg_specification)

    args = parser.parse_args()
    kwargs = vars(args)
    del(kwargs['operation'])    # Remove since this is not operation input.
    operation(**kwargs)

if __name__ == '__main__':
    run_model()