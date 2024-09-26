from typing import Generator

import numpy as np
import h5py

from util.hdf5 import IMAGES_KEY, LABELS_KEY


class HDF5DatasetGenerator:
    """Generator class for providing the dataset to a model."""

    SHUFFLE_SEED = 2024
    BINARIZATION_THRESHOLD = 0.5

    num_images: int
    batch_size: int

    images: np.array
    labels: np.array

    shuffle: bool
    binarize_labels: bool

    def __init__(
        self,
        data_file_path: str,
        batch_size: int,
        shuffle: bool,
        binarize_labels: bool,
        data_augmentor
    ):
        self.batch_size = batch_size
        self.binarize_labels = binarize_labels
        self.shuffle = shuffle

        data_file = h5py.File(data_file_path, 'r+')
        # Copy specifically such that we can keep this in memory
        self.images = np.copy(data_file[IMAGES_KEY])
        self.labels = np.copy(data_file[LABELS_KEY])
        self.num_images = len(self.images)
        data_file.close()

        self.data_augmentor = data_augmentor
        
    def __call__(self, passes=np.inf) -> Generator:
        """Generate data for a certain number of passes. By default, this is infinite passes."""
        pass_idx = 0
        while pass_idx < passes:
            for batch_idx in np.arange(0, self.num_images, self.batch_size):
                images = self.images[batch_idx: batch_idx + self.batch_size]
                labels = self.labels[batch_idx: batch_idx + self.batch_size]

                if self.data_augmentor is not None:
                    image_generator = self.data_augmentor.flow(images, seed=self.SHUFFLE_SEED, batch_size=self.batch_size, shuffle=self.shuffle)
                    label_generator = self.data_augmentor.flow(labels, seed=self.SHUFFLE_SEED, batch_size=self.batch_size, shuffle=self.shuffle)

                    train_generator = zip(image_generator, label_generator)
                    images, labels = next(train_generator)

                if self.binarize_labels:
                    labels[labels > self.BINARIZATION_THRESHOLD] = 1.
                    labels[labels <= self.BINARIZATION_THRESHOLD] = 0.
        
                yield images, labels

            pass_idx += 1
