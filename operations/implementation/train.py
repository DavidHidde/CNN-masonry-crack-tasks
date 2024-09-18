import os
from typing import Any, Union

from network.loss import determine_loss_function
from network.metrics import get_standard_metrics
from network.model import build_model, load_model
from network.optimizer import determine_optimizer
from operations.operation import Operation
import operations.arguments as arguments
from network.callbacks import EpochCheckpoint, TrainingMonitor
from util.hdf5 import HDF5DatasetGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from util.config import load_network_config, load_data_config, load_output_config
from util.types import MetricType


class Train(Operation):
    """Train a model on a dataset."""

    def __call__(
        self,
        dataset: str,
        network: str,
        weights: Union[str, None],
        save_model: bool,
        checkpoint_epochs: int,
        monitor_metric: MetricType
    ) -> None:
        """Start training a model on a dataset."""
        network_config = load_network_config(network)
        dataset_config = load_data_config(dataset)
        output_config = load_output_config(network_id=network_config.id, dataset_id=dataset_config.dataset_dir)

        # %%
        # Prepare model for training. Load one if we have it and otherwise start again
        #
        if weights is not None:
            model = load_model(network_config, output_config, dataset_config.image_dims, weights)

            # Determine start epoch. We assume the file follows the regular naming scheme (epoch_X_*.)
            parts = weights.split('_')
            for idx, part in enumerate(parts):
                if part == 'epoch':
                    start_epoch = int(parts[idx + 1])
                    break
            else:
                raise ValueError('Could not determine start epoch from weights file.')
        else:
            model = build_model(network_config, dataset_config.image_dims)
            start_epoch = 0

        model.compile(
            optimizer=determine_optimizer(network_config),
            loss=determine_loss_function(network_config),
            metrics=[get_standard_metrics()]
        )

        # Data augmentation for training and validation sets
        if network_config.augment_data:
            data_augmentor = ImageDataGenerator(
                rotation_range=20,
                zoom_range=0.15,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            data_augmentor = None

        # Load data generators
        train_gen = HDF5DatasetGenerator(
            output_config.train_set_file,
            network_config.batch_size,
            False,
            network_config.binarize_labels,
            data_augmentor
        )
        val_gen = HDF5DatasetGenerator(
            output_config.validation_set_file,
            network_config.batch_size,
            False,
            network_config.binarize_labels,
            None
        )

        # Serialize model to JSON
        try:
            model_json = model.to_json()
            with open(output_config.model_file, 'w') as json_file:
                json_file.write(model_json)
        except:
            print('Warning: Unable to write model.json!!')

        # Setup all callbacks
        csv_logger = CSVLogger(output_config.log_file, append=True, separator=';')
        epoch_checkpoint = EpochCheckpoint(
            output_config.checkpoints_dir,
            not save_model,
            checkpoint_epochs,
            start_epoch
        )
        training_monitor = TrainingMonitor(
            output_config.progression_file,
            output_config.metrics_file,
            start_epoch,
            monitor_metric
        )
        model_checkpoint = ModelCheckpoint(
            os.path.join(output_config.best_models_dir, f'epoch_{{epoch}}_{monitor_metric.value}_{{val_{monitor_metric.value}:.3f}}.keras'),
            monitor=f'val_{monitor_metric.value}',
            verbose=1,
            save_best_only=True,
            mode='max',
            save_weights_only=not save_model
        )

        # %%
        # Train the network
        #
        H = model.fit(
            train_gen(),
            steps_per_epoch=train_gen.num_images // network_config.batch_size,
            validation_data=val_gen(),
            validation_steps=val_gen.num_images // network_config.batch_size,
            initial_epoch=start_epoch,
            epochs=network_config.num_epochs,
            max_queue_size=network_config.batch_size * 2,
            callbacks=[
                csv_logger,
                epoch_checkpoint,
                training_monitor,
                model_checkpoint
            ],
            verbose=1
        )

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.DATASET_ARGUMENT,
            arguments.NETWORK_ARGUMENT,
            arguments.WEIGHTS_FILE_ARGUMENT,
            arguments.SAVE_MODEL_ARGUMENT,
            arguments.EPOCHS_PER_CHECKPOINT_ARGUMENT,
            arguments.MONITOR_METRIC_ARGUMENT
        ]
