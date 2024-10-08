import cv2
import numpy as np
import progressbar
from imutils import paths
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from operations.operation import Operation
import operations.arguments as arguments
from typing import Any
from util.hdf5 import HDF5DatasetWriter
from util.config import load_data_config, load_output_config


class Build(Operation):
    """Build the config."""

    def __call__(self, dataset: str, no_labels: bool) -> None:
        """Process the dataset into a training and validation set."""
        dataset_config = load_data_config(dataset)
        output_config = load_output_config(dataset_id=dataset_config.dataset_dir)

        # Grab the paths to the images and masks and sort them to ensure everything is properly in order.
        img_paths = list(paths.list_images(output_config.images_dir))
        label_paths = list(paths.list_images(output_config.labels_dir)) if no_labels is False else img_paths
        img_paths.sort()
        label_paths.sort()

        if dataset_config.validation_split_percent == 0.:  # Only training data
            datasets = [(list(zip(img_paths, label_paths)), output_config.train_set_file)]
        elif dataset_config.validation_split_percent == 1. or no_labels:  # Only validation data
            datasets = [(list(zip(img_paths, label_paths)), output_config.validation_set_file)]
        else:  # Perform stratified sampling from the training set to build the testing split from the training data
            train_img_split, val_img_split, train_label_split, val_label_split = train_test_split(
                img_paths,
                label_paths,
                test_size=dataset_config.validation_split_percent,
                random_state=42
            )

            # Construct a list pairing the training, validation, and testing image paths along with their corresponding labels
            # and output HDF5 files
            datasets = [
                (list(zip(train_img_split, train_label_split)), output_config.train_set_file),
                (list(zip(val_img_split, val_label_split)), output_config.validation_set_file)
            ]

        # Loop over the dataset tuples
        for (data_pairs, output_file) in datasets:
            print("[INFO] building {}...".format(output_file))

            # Create HDF5 writer
            writer = HDF5DatasetWriter((len(data_pairs), *dataset_config.image_dims), output_file)

            # initialize the progress bar
            widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=len(data_pairs), widgets=widgets).start()

            # loop over the image paths
            for (idx, (im_path, label_path)) in enumerate(data_pairs):
                # Load the image and label and resize them if necessary
                image = cv2.imread(im_path)

                if no_labels:
                    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                else:
                    label = cv2.imread(label_path, 0)

                if dataset_config.image_dims != image.shape:
                    image = resize(image, dataset_config.image_dims, mode='constant', preserve_range=True)
                if dataset_config.image_dims[:2] != label.shape:
                    label = resize(label, dataset_config.image_dims[:2], mode='constant', preserve_range=True)

                # Normalize intensity values: [0,1]
                label = np.expand_dims(label, axis=-1)
                label = label / 255
                image = image / 255

                # Add to the HDF5 dataset and update the progress bar
                writer.add([image], [label])
                pbar.update(idx)

            # Close the progress bar and the HDF5 writer
            pbar.finish()
            writer.close()

    def get_cli_arguments(self) -> list[dict[str, Any]]:
        return [
            arguments.DATASET_ARGUMENT,
            arguments.NO_LABELS_ARGUMENT,
        ]
