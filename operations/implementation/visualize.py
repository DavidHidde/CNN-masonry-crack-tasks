from typing import Any

from operations.operation import Operation
import operations.arguments as arguments
from util.config import load_network_config, load_output_config, load_data_config
from keras.utils import plot_model
from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network.model import build_model
from network.optimizer import determine_optimizer
from contextlib import redirect_stdout

class Visualize(Operation):
    """Visualize the model architecture."""

    def __call__(self, network: str, dataset: str) -> None:
        network_config = load_network_config(network)
        dataset_config = load_data_config(dataset)
        output_config = load_output_config(network_id=network_config.id, dataset_id=dataset_config.dataset_dir)

        # %%
        # Prepare model for training
        #
        model = build_model(network_config, dataset_config.image_dims)
        model.compile(
            optimizer=determine_optimizer(network_config),
            loss=determine_loss_function(network_config),
            metrics=[get_standard_metrics()]
        )

        # Create a visual plot
        plot_model(model, to_file=output_config.figure_file, show_shapes=True)

        # Create a txt summary
        with open(output_config.txt_summary_file, 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.NETWORK_ARGUMENT,
            arguments.DATASET_ARGUMENT
        ]

