from typing import Any, Union

from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network.optimizer import determine_optimizer
from operations.operation import Operation
import operations.arguments as arguments
from network.model import load_model
from util.hdf5 import HDF5DatasetGenerator
from util.visualize import visualize_prediction_comparisons, save_predictions
from util.config import load_network_config, load_data_config, load_output_config


class Test(Operation):
    """Operation which tests a trained network"""

    def __call__(self, dataset: str, network: str, weights: Union[str, None], dilate: bool, visualize_comparisons: bool) -> None:
        """Generate the predictions given a specific configuration."""
        network_config = load_network_config(network)
        dataset_config = load_data_config(dataset)
        output_config = load_output_config(network_id=network_config.id, dataset_id=dataset_config.dataset_dir)

        model = load_model(network_config, output_config, dataset_config.image_dims, weights)
        model.compile(
            optimizer=determine_optimizer(network_config),
            loss=determine_loss_function(network_config),
            metrics=[get_standard_metrics()]
        )

        # Do not use data augmentation when evaluating model: aug=None
        eval_gen = HDF5DatasetGenerator(
            output_config.validation_set_file,
            network_config.batch_size,
            False,
            network_config.binarize_labels,
            None
        )

        # Use the pretrained model to generate predictions for the input samples from a data generator
        predictions = model.predict(
            eval_gen(),
            steps=eval_gen.num_images // network_config.batch_size + 1,
            max_queue_size=network_config.batch_size * 2,
            verbose=1
        )
        metrics = model.evaluate(
            eval_gen(),
            steps=eval_gen.num_images // network_config.batch_size + 1,
            verbose=1,
            return_dict=True
        )
        print('\n--- Results ---')
        for key, value in metrics.items():
            print(f'{key}: {(value * 100):.2f}')
        print('---------------\n')

        # Plot or just save the output
        if visualize_comparisons:
            visualize_prediction_comparisons(predictions, output_config, dilate)
        else:
            save_predictions(predictions, output_config)

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.NETWORK_ARGUMENT,
            arguments.DATASET_ARGUMENT,
            arguments.WEIGHTS_FILE_ARGUMENT,
            arguments.DILATE_VALIDATION_LABELS_ARGUMENT,
            arguments.VISUALIZE_COMPARISONS_ARGUMENT,
        ]
